"""Tests for agent.events telemetry: event sequence, identity, switches and the drop path."""

from __future__ import annotations

import asyncio
from datetime import datetime
import gc
import hashlib
import hmac
import io
import json
import logging
from pathlib import Path
import re
import socket
import time
from typing import Any, Dict, List, Optional, Sequence
import uuid

import httpx
import pytest

from app.agent.orchestrator import AgentOrchestrator
from app.agent.streaming import stream_agent_turn
from app.config import Settings
from app.llm.base import LLMClientError, LLMResponse, ToolCallRequest, parse_openai_chat_response
from app.llm.streaming import ChatStreamAssembler
from app.logging_config import LOG_FORMAT, RequestIdFormatter
from app.main import create_app
from app.mcp.client import MCPClientError
from app.schemas import ChatResponse
from app.telemetry import build_telemetry, missing_telemetry_settings
from app.telemetry.events import (
    clean_app_version,
    clean_client,
    clean_session_id,
    user_key,
)
from app.telemetry.producer import (
    DROP_DELIVERY_FAILED,
    DROP_KAFKA_UNAVAILABLE,
    DROP_QUEUE_FULL,
    DROP_SEND_FAILED,
    KafkaTelemetry,
)
from app.telemetry.request import RequestTelemetry

SALT = "test-telemetry-salt"
CLIENT_IP = "203.0.113.77"
PROMPT = "SECRET-PROMPT-TEXT: how many wins does Ole Miss have?"
SESSION_ID = "3F2504E0-4F89-41D3-9A0C-0305E82C3301"
UI_HEADERS = {
    "X-Real-IP": CLIENT_IP,
    "X-Session-Id": SESSION_ID,
    "X-Client": "web",
    "X-App-Version": "a1b2c3d",
}

CONTRACT_FIELDS = {
    "event_id",
    "event_type",
    "request_id",
    "session_id",
    "user_key",
    "event_time",
    "tool",
    "latency_ms",
    "status",
    "error_code",
    "model",
    "input_tokens",
    "output_tokens",
    "prompt_chars",
    "client",
    "app_version",
}
COMMON = {"event_id", "event_type", "request_id", "session_id", "user_key", "event_time"}
STARTED_FIELDS = COMMON | {"prompt_chars", "client", "app_version"}
TOOL_FIELDS = COMMON | {"tool", "latency_ms", "status", "error_code"}
TERMINAL_FIELDS = COMMON | {
    "latency_ms",
    "status",
    "error_code",
    "model",
    "input_tokens",
    "output_tokens",
    "prompt_chars",
    "client",
    "app_version",
}
RFC3339_UTC = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z")


# --- fakes -------------------------------------------------------------------------------


class FakeProducer:
    """Stands in for AIOKafkaProducer: records what would have been sent."""

    def __init__(
        self,
        start_error: Optional[BaseException] = None,
        hang_on_start: bool = False,
        send_error: Optional[BaseException] = None,
        delivery_error: Optional[BaseException] = None,
    ) -> None:
        self.start_error = start_error
        self.hang_on_start = hang_on_start
        self.send_error = send_error
        self.delivery_error = delivery_error
        self.messages: List[Dict[str, Any]] = []
        self.started = False
        self.stopped = False

    async def start(self) -> None:
        if self.hang_on_start:
            await asyncio.Event().wait()
        if self.start_error:
            raise self.start_error
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def send(self, topic: str, value: bytes, key: Optional[bytes] = None) -> "asyncio.Future[Any]":
        if self.send_error:
            raise self.send_error
        self.messages.append({"topic": topic, "key": key, "value": value})
        delivery: "asyncio.Future[Any]" = asyncio.get_running_loop().create_future()
        if self.delivery_error:
            delivery.set_exception(self.delivery_error)
        else:
            delivery.set_result(None)
        return delivery

    def events(self) -> List[Dict[str, Any]]:
        return [json.loads(m["value"]) for m in self.messages]


class ScriptedRouter:
    """LLM router stand-in that plays back responses or raises errors."""

    def __init__(self, script: Sequence[Any]) -> None:
        self.script = list(script)

    def _next(self) -> LLMResponse:
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    async def chat(self, messages: Any, mcp_tools: Any = None, provider: Any = None) -> LLMResponse:
        return self._next()

    async def chat_stream(
        self,
        messages: Any,
        on_text_delta: Any,
        mcp_tools: Any = None,
        provider: Any = None,
        allow_fallback: bool = True,
    ) -> LLMResponse:
        response = self._next()
        if response.content and not response.tool_calls:
            on_text_delta(response.content)
        return response


class FakeMCP:
    def __init__(self, failing: Sequence[str] = (), timeout: Sequence[str] = ()) -> None:
        self.failing = set(failing)
        self.timeout = set(timeout)
        self.calls: List[str] = []

    async def list_tools(self, fallback_to_defaults: bool = True) -> List[Dict[str, Any]]:
        return []

    async def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> str:
        self.calls.append(name)
        if name in self.timeout:
            try:
                raise httpx.ReadTimeout("slow")
            except httpx.ReadTimeout as exc:
                raise MCPClientError("Timed out") from exc
        if name in self.failing:
            raise MCPClientError("upstream down")
        return "[]"


def tool_turn(*names: str, tokens: tuple = (100, 10)) -> LLMResponse:
    return LLMResponse(
        content="",
        tool_calls=[ToolCallRequest(id=f"call_{i}", name=n, arguments={"team": "Ole Miss"}) for i, n in enumerate(names)],
        provider_used="llama",
        input_tokens=tokens[0],
        output_tokens=tokens[1],
    )


def final_turn(text: str = "Ole Miss has 10 wins.", provider: str = "llama", tokens: tuple = (200, 20)) -> LLMResponse:
    return LLMResponse(content=text, provider_used=provider, input_tokens=tokens[0], output_tokens=tokens[1])


@pytest.fixture()
def telemetry_settings(sample_skills_dir: Path) -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        llama_model="test-llama-model",
        gemini_model="gemini-test",
        gemini_api_key="test-gemini-key",
        skills_dir=str(sample_skills_dir),
        kafka_bootstrap="fake-kafka:9092",
        telemetry_salt=SALT,
        log_level="DEBUG",
    )


def make_app(settings: Settings, router: Any, mcp: Any, producer: FakeProducer) -> Any:
    app = create_app(settings)
    app.state.orchestrator = AgentOrchestrator(
        mcp_client=mcp, llm_router=router, skills_loader=app.state.skills_loader
    )
    app.state.telemetry = KafkaTelemetry(settings.kafka_bootstrap or "", producer_factory=lambda _: producer)
    return app


async def send(app: Any, method: str, path: str, **kwargs: Any) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        return await client.request(method, path, **kwargs)


async def with_telemetry(app: Any, method: str, path: str, **kwargs: Any) -> httpx.Response:
    """Run one request with the app's telemetry started, then flush it."""
    telemetry: KafkaTelemetry = app.state.telemetry
    await telemetry.start()
    try:
        return await send(app, method, path, **kwargs)
    finally:
        await telemetry.stop()


def stream_body(prompt: str = PROMPT) -> Dict[str, Any]:
    return {"messages": [{"role": "user", "content": prompt}]}


def assert_contract(events: List[Dict[str, Any]]) -> None:
    expected = {
        "request_started": STARTED_FIELDS,
        "tool_called": TOOL_FIELDS,
        "request_completed": TERMINAL_FIELDS,
        "request_failed": TERMINAL_FIELDS,
    }
    for event in events:
        assert set(event) == expected[event["event_type"]]
        assert set(event) <= CONTRACT_FIELDS
        assert uuid.UUID(event["event_id"]).version == 4
        assert RFC3339_UTC.fullmatch(event["event_time"]), event["event_time"]
    assert len({e["event_id"] for e in events}) == len(events)
    assert len({e["request_id"] for e in events}) == 1


# --- event sequence -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_emits_started_tools_and_completed(telemetry_settings: Settings) -> None:
    producer = FakeProducer()
    mcp = FakeMCP()
    router = ScriptedRouter([tool_turn("get_cfb_scores", "get_team_schedule"), final_turn()])
    app = make_app(telemetry_settings, router, mcp, producer)

    response = await with_telemetry(app, "POST", "/chat", json={"message": PROMPT}, headers=UI_HEADERS)

    assert response.status_code == 200
    events = producer.events()
    assert [e["event_type"] for e in events] == [
        "request_started",
        "tool_called",
        "tool_called",
        "request_completed",
    ]
    assert_contract(events)

    request_id = response.headers["X-Request-Id"]
    assert uuid.UUID(request_id).version == 4
    assert all(e["request_id"] == request_id for e in events)
    assert all(m["key"] == request_id.encode() for m in producer.messages)
    assert all(m["topic"] == "agent.events" for m in producer.messages)

    expected_key = hmac.new(SALT.encode(), CLIENT_IP.encode(), hashlib.sha256).hexdigest()
    assert all(e["user_key"] == expected_key for e in events)
    assert all(e["session_id"] == SESSION_ID.lower() for e in events)

    started, tool1, tool2, completed = events
    assert started["prompt_chars"] == len(PROMPT)
    assert started["client"] == "web" and started["app_version"] == "a1b2c3d"
    assert [tool1["tool"], tool2["tool"]] == ["get_cfb_scores", "get_team_schedule"]
    for tool in (tool1, tool2):
        assert tool["status"] == "ok" and tool["error_code"] is None
        assert isinstance(tool["latency_ms"], int) and tool["latency_ms"] >= 0
    assert completed["status"] == "completed"
    assert completed["error_code"] is None
    assert completed["model"] == "test-llama-model"
    assert completed["input_tokens"] == 300 and completed["output_tokens"] == 30
    assert completed["prompt_chars"] == len(PROMPT)
    assert completed["client"] == "web" and completed["app_version"] == "a1b2c3d"
    assert isinstance(completed["latency_ms"], int)
    assert started["event_time"] <= tool1["event_time"] <= completed["event_time"]


@pytest.mark.asyncio
async def test_tool_error_is_reported_and_request_still_completes(telemetry_settings: Settings) -> None:
    producer = FakeProducer()
    mcp = FakeMCP(failing=["get_team_schedule"], timeout=["get_cfb_rankings"])
    router = ScriptedRouter(
        [tool_turn("get_cfb_scores", "get_team_schedule", "get_cfb_rankings"), final_turn(provider="gemini")]
    )
    app = make_app(telemetry_settings, router, mcp, producer)

    response = await with_telemetry(app, "POST", "/chat", json={"message": PROMPT}, headers=UI_HEADERS)

    assert response.status_code == 200
    events = producer.events()
    assert_contract(events)
    tools = [e for e in events if e["event_type"] == "tool_called"]
    assert [(t["tool"], t["status"], t["error_code"]) for t in tools] == [
        ("get_cfb_scores", "ok", None),
        ("get_team_schedule", "error", "mcp_error"),
        ("get_cfb_rankings", "error", "mcp_timeout"),
    ]
    assert events[-1]["event_type"] == "request_completed"
    assert events[-1]["model"] == "gemini-test"


@pytest.mark.asyncio
async def test_llm_failure_emits_request_failed(telemetry_settings: Settings) -> None:
    producer = FakeProducer()
    router = ScriptedRouter([tool_turn("get_cfb_scores"), LLMClientError("both providers failed: detail")])
    app = make_app(telemetry_settings, router, FakeMCP(), producer)

    response = await with_telemetry(app, "POST", "/chat", json={"message": PROMPT}, headers=UI_HEADERS)

    assert response.status_code == 502
    events = producer.events()
    assert_contract(events)
    assert [e["event_type"] for e in events] == ["request_started", "tool_called", "request_failed"]
    failed = events[-1]
    assert failed["status"] == "failed"
    assert failed["error_code"] == "llm_error"
    # Usage from the turns that did succeed is kept.
    assert failed["model"] == "test-llama-model"
    assert failed["input_tokens"] == 100 and failed["output_tokens"] == 10
    assert "detail" not in json.dumps(events)


@pytest.mark.asyncio
async def test_unexpected_error_emits_internal_error(telemetry_settings: Settings) -> None:
    class BrokenAgent:
        async def run(self, **kwargs: Any) -> ChatResponse:
            raise RuntimeError("boom")

    producer = FakeProducer()
    app = make_app(telemetry_settings, ScriptedRouter([]), FakeMCP(), producer)
    app.state.orchestrator = BrokenAgent()

    with pytest.raises(RuntimeError):
        await with_telemetry(app, "POST", "/chat", json={"message": PROMPT}, headers=UI_HEADERS)

    events = producer.events()
    assert [e["event_type"] for e in events] == ["request_started", "request_failed"]
    assert events[-1]["error_code"] == "internal_error"


@pytest.mark.asyncio
async def test_stream_completes_when_the_stream_ends(telemetry_settings: Settings) -> None:
    producer = FakeProducer()
    router = ScriptedRouter([tool_turn("get_cfb_scores"), final_turn()])
    app = make_app(telemetry_settings, router, FakeMCP(), producer)

    response = await with_telemetry(app, "POST", "/chat/stream", json=stream_body(), headers=UI_HEADERS)

    assert response.status_code == 200
    assert response.text.rstrip().endswith("event: done\ndata: {}")
    events = producer.events()
    assert_contract(events)
    assert [e["event_type"] for e in events] == ["request_started", "tool_called", "request_completed"]
    assert events[0]["request_id"] == response.headers["X-Request-Id"]
    assert events[-1]["input_tokens"] == 300


@pytest.mark.asyncio
async def test_stream_llm_failure_emits_request_failed(telemetry_settings: Settings) -> None:
    producer = FakeProducer()
    router = ScriptedRouter([LLMClientError("Gemini returned HTTP 400: secret detail")])
    app = make_app(telemetry_settings, router, FakeMCP(), producer)

    response = await with_telemetry(app, "POST", "/chat/stream", json=stream_body(), headers=UI_HEADERS)

    assert response.status_code == 200
    events = producer.events()
    assert [e["event_type"] for e in events] == ["request_started", "request_failed"]
    assert events[-1]["error_code"] == "llm_error"
    assert events[-1]["model"] is None and events[-1]["input_tokens"] is None


def make_recorder(sink: Any) -> RequestTelemetry:
    return RequestTelemetry(
        sink=sink,
        request_id=str(uuid.uuid4()),
        user_key=user_key(SALT, CLIENT_IP),
        session_id=None,
        client="web",
        app_version=None,
        prompt_chars=2,
    )


class ListSink:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def emit(self, event: Dict[str, Any]) -> None:
        self.events.append(dict(event))


@pytest.mark.asyncio
async def test_stream_abandoned_mid_turn_is_client_disconnected() -> None:
    started = asyncio.Event()
    sink = ListSink()

    class HangingAgent:
        async def run(self, on_tool_event: Any = None, **kwargs: Any) -> ChatResponse:
            on_tool_event("tool.start", {"id": "1:a", "name": "get_cfb_scores", "args": {}})
            started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    stream = stream_agent_turn(HangingAgent(), user_message="Hi", telemetry=make_recorder(sink))
    assert (await stream.__anext__()).startswith("event: tool.start")
    await started.wait()
    await stream.aclose()

    assert [e["event_type"] for e in sink.events] == ["request_started", "request_failed"]
    assert sink.events[-1]["error_code"] == "client_disconnected"
    assert sink.events[-1]["status"] == "failed"


@pytest.mark.asyncio
async def test_stream_abandoned_after_the_turn_finished_is_still_a_disconnect() -> None:
    sink = ListSink()

    class QuickAgent:
        async def run(self, on_text_delta: Any = None, **kwargs: Any) -> ChatResponse:
            on_text_delta("Hello")
            return ChatResponse(reply="Hello", provider_used="llama")

    stream = stream_agent_turn(QuickAgent(), user_message="Hi", telemetry=make_recorder(sink))
    assert (await stream.__anext__()).startswith("event: message.delta")
    await stream.aclose()  # "done" was never delivered

    assert sink.events[-1]["event_type"] == "request_failed"
    assert sink.events[-1]["error_code"] == "client_disconnected"


@pytest.mark.asyncio
async def test_asgi_client_disconnect_mid_stream_is_reported(telemetry_settings: Settings) -> None:
    """Drive the app at the ASGI level and disconnect after the first SSE frame."""
    producer = FakeProducer()
    started = asyncio.Event()

    class HangingAgent:
        async def run(self, on_tool_event: Any = None, **kwargs: Any) -> ChatResponse:
            on_tool_event("tool.start", {"id": "1:a", "name": "get_cfb_scores", "args": {}})
            started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    app = make_app(telemetry_settings, ScriptedRouter([]), FakeMCP(), producer)
    app.state.orchestrator = HangingAgent()
    telemetry: KafkaTelemetry = app.state.telemetry
    await telemetry.start()

    body = json.dumps(stream_body()).encode()
    first_frame = asyncio.Event()
    request_sent = False
    sent: List[Dict[str, Any]] = []

    async def receive() -> Dict[str, Any]:
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        await first_frame.wait()
        return {"type": "http.disconnect"}

    async def send_message(message: Dict[str, Any]) -> None:
        sent.append(message)
        if message["type"] == "http.response.body" and message.get("body"):
            first_frame.set()

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/chat/stream",
        "raw_path": b"/chat/stream",
        "root_path": "",
        "query_string": b"",
        "headers": [(b"content-type", b"application/json"), (b"host", b"testserver")]
        + [(k.lower().encode(), v.encode()) for k, v in UI_HEADERS.items()],
        "client": ("10.0.0.5", 50000),
        "server": ("testserver", 80),
    }
    await asyncio.wait_for(app(scope, receive, send_message), timeout=5)
    for _ in range(20):  # let an abandoned generator be finalized
        gc.collect()
        await asyncio.sleep(0.01)
        if any(e["event_type"] == "request_failed" for e in producer.events()):
            break
    await telemetry.stop()

    assert started.is_set()
    events = producer.events()
    assert [e["event_type"] for e in events] == ["request_started", "request_failed"]
    assert events[-1]["error_code"] == "client_disconnected"
    start_message = next(m for m in sent if m["type"] == "http.response.start")
    headers = dict(start_message["headers"])
    assert headers[b"x-request-id"].decode() == events[0]["request_id"]


@pytest.mark.asyncio
async def test_rate_limited_requests_emit_no_events(telemetry_settings: Settings) -> None:
    settings = telemetry_settings.model_copy(update={"rate_limit_requests": 1})
    producer = FakeProducer()
    app = make_app(settings, ScriptedRouter([final_turn()]), FakeMCP(), producer)

    telemetry: KafkaTelemetry = app.state.telemetry
    await telemetry.start()
    first = await send(app, "POST", "/chat", json={"message": PROMPT}, headers=UI_HEADERS)
    second = await send(app, "POST", "/chat", json={"message": PROMPT}, headers=UI_HEADERS)
    await telemetry.stop()

    assert first.status_code == 200 and second.status_code == 429
    assert "X-Request-Id" in second.headers
    assert [e["event_type"] for e in producer.events()] == ["request_started", "request_completed"]


# --- privacy ------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_event_contains_prompt_text_or_ip(telemetry_settings: Settings) -> None:
    producer = FakeProducer()
    router = ScriptedRouter(
        [tool_turn("get_cfb_scores"), final_turn(), tool_turn("get_cfb_rankings"), final_turn()]
    )
    app = make_app(telemetry_settings, router, FakeMCP(), producer)
    telemetry: KafkaTelemetry = app.state.telemetry
    await telemetry.start()
    await send(app, "POST", "/chat", json={"message": PROMPT}, headers=UI_HEADERS)
    await send(
        app,
        "POST",
        "/chat/stream",
        json={"messages": [{"role": "user", "content": "earlier " + PROMPT}, {"role": "assistant", "content": "ok"}, {"role": "user", "content": PROMPT}]},
        headers=UI_HEADERS,
    )
    await telemetry.stop()

    assert len(producer.messages) == 6
    raw = b"\n".join(m["value"] + (m["key"] or b"") for m in producer.messages)
    assert b"SECRET-PROMPT-TEXT" not in raw
    assert b"Ole Miss" not in raw
    assert CLIENT_IP.encode() not in raw
    assert hashlib.sha256(CLIENT_IP.encode()).hexdigest().encode() not in raw  # never an unsalted hash
    assert SALT.encode() not in raw


# --- headers and identity -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("3f2504e0-4f89-41d3-9a0c-0305e82c3301", "3f2504e0-4f89-41d3-9a0c-0305e82c3301"),
        ("3F2504E0-4F89-41D3-9A0C-0305E82C3301", "3f2504e0-4f89-41d3-9a0c-0305e82c3301"),
        (" 3f2504e0-4f89-41d3-9a0c-0305e82c3301 ", "3f2504e0-4f89-41d3-9a0c-0305e82c3301"),
        ("3f2504e04f8941d39a0c0305e82c3301", None),
        ("{3f2504e0-4f89-41d3-9a0c-0305e82c3301}", None),
        ("urn:uuid:3f2504e0-4f89-41d3-9a0c-0305e82c3301", None),
        ("3f2504e0-4f89-41d3-9a0c-0305e82c3301x", None),
        ("user@example.com", None),
        ("", None),
        (None, None),
    ],
)
def test_session_id_must_be_a_uuid(value: Optional[str], expected: Optional[str]) -> None:
    assert clean_session_id(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("web", "web"),
        ("ios-app_2.x", "ios-app_2.x"),
        ("a" * 32, "a" * 32),
        ("a" * 33, None),
        ("Web", None),
        ("web app", None),
        ("web;drop", None),
        ("", None),
        (None, None),
    ],
)
def test_client_header_validation(value: Optional[str], expected: Optional[str]) -> None:
    assert clean_client(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("a1b2c3d", "a1b2c3d"),
        ("1.4.0+build.7", "1.4.0+build.7"),
        ("dev", "dev"),
        ("V" * 64, "V" * 64),
        ("V" * 65, None),
        ("1.0 beta", None),
        ("1.0/2", None),
        ("", None),
        (None, None),
    ],
)
def test_app_version_header_validation(value: Optional[str], expected: Optional[str]) -> None:
    assert clean_app_version(value) == expected


@pytest.mark.asyncio
async def test_malformed_headers_are_stored_as_null(telemetry_settings: Settings) -> None:
    producer = FakeProducer()
    app = make_app(telemetry_settings, ScriptedRouter([final_turn()]), FakeMCP(), producer)
    headers = {
        "X-Real-IP": CLIENT_IP,
        "X-Session-Id": "not-a-uuid",
        "X-Client": "Mozilla/5.0",
        "X-App-Version": "1.0 <script>",
    }

    await with_telemetry(app, "POST", "/chat", json={"message": PROMPT}, headers=headers)

    for event in producer.events():
        assert event["session_id"] is None
        if "client" in event:
            assert event["client"] is None and event["app_version"] is None


@pytest.mark.asyncio
async def test_missing_headers_are_null(telemetry_settings: Settings) -> None:
    producer = FakeProducer()
    app = make_app(telemetry_settings, ScriptedRouter([final_turn()]), FakeMCP(), producer)

    await with_telemetry(app, "POST", "/chat", json={"message": PROMPT})

    started = producer.events()[0]
    assert started["session_id"] is None and started["client"] is None and started["app_version"] is None
    # Without X-Real-IP the rate limiter's fallback (the socket peer) is hashed, the same way.
    assert started["user_key"] == user_key(SALT, "127.0.0.1")


def test_user_key_is_hmac_sha256_of_the_ip() -> None:
    key = user_key(SALT, CLIENT_IP)
    assert key == hmac.new(SALT.encode(), CLIENT_IP.encode(), hashlib.sha256).hexdigest()
    assert re.fullmatch(r"[0-9a-f]{64}", key)
    assert key != hashlib.sha256((CLIENT_IP + SALT).encode()).hexdigest()
    assert key != hashlib.sha256(CLIENT_IP.encode()).hexdigest()
    assert user_key("other-salt", CLIENT_IP) != key
    assert user_key(SALT, "203.0.113.78") != key


def test_salt_whitespace_is_ignored(sample_skills_dir: Path) -> None:
    settings = Settings(_env_file=None, kafka_bootstrap="k:9092", telemetry_salt=f"  {SALT}\n")  # type: ignore[call-arg]
    assert settings.telemetry_key == SALT
    assert SALT not in repr(settings)


# --- request id ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_every_response_carries_a_fresh_request_id(test_settings: Settings) -> None:
    app = create_app(test_settings)
    first = await send(app, "GET", "/rate-limit", headers={"X-Request-Id": "client-chosen"})
    second = await send(app, "GET", "/skills")
    ids = [first.headers["X-Request-Id"], second.headers["X-Request-Id"]]
    assert all(uuid.UUID(i).version == 4 for i in ids)
    assert ids[0] != ids[1]
    assert "client-chosen" not in ids


@pytest.mark.asyncio
async def test_request_id_is_in_every_log_line_of_the_request(telemetry_settings: Settings) -> None:
    producer = FakeProducer()
    router = ScriptedRouter([tool_turn("get_cfb_scores"), final_turn(), final_turn()])
    app = make_app(telemetry_settings, router, FakeMCP(), producer)

    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    handler.setFormatter(RequestIdFormatter(LOG_FORMAT))
    app_logger = logging.getLogger("sportsagentic")
    app_logger.addHandler(handler)
    try:
        chat = await with_telemetry(app, "POST", "/chat", json={"message": PROMPT}, headers=UI_HEADERS)
        stream = await with_telemetry(app, "POST", "/chat/stream", json=stream_body(), headers=UI_HEADERS)
    finally:
        app_logger.removeHandler(handler)

    lines = buffer.getvalue().splitlines()
    for response, path in ((chat, "POST /chat ->"), (stream, "POST /chat/stream ->")):
        request_id = response.headers["X-Request-Id"]
        tagged = [line for line in lines if f"| {request_id} |" in line]
        assert any(path in line for line in tagged)
        assert any("Agent completed turn" in line for line in tagged)
    telemetry_lines = [line for line in lines if "Telemetry" in line]
    assert telemetry_lines and all("| - |" in line for line in telemetry_lines)


# --- switches -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("bootstrap", "salt", "missing"),
    [
        (None, SALT, ["KAFKA_BOOTSTRAP"]),
        ("k:9092", None, ["TELEMETRY_SALT"]),
        ("   ", SALT, ["KAFKA_BOOTSTRAP"]),
        ("k:9092", "  ", ["TELEMETRY_SALT"]),
        (None, None, ["KAFKA_BOOTSTRAP", "TELEMETRY_SALT"]),
    ],
)
def test_telemetry_is_off_unless_bootstrap_and_salt_are_set(
    bootstrap: Optional[str], salt: Optional[str], missing: List[str]
) -> None:
    settings = Settings(_env_file=None, kafka_bootstrap=bootstrap, telemetry_salt=salt)  # type: ignore[call-arg]
    assert missing_telemetry_settings(settings) == missing
    assert build_telemetry(settings) is None


def test_telemetry_is_on_with_bootstrap_and_salt() -> None:
    settings = Settings(_env_file=None, kafka_bootstrap="k:9092", telemetry_salt=SALT)  # type: ignore[call-arg]
    assert missing_telemetry_settings(settings) == []
    telemetry = build_telemetry(settings)
    assert isinstance(telemetry, KafkaTelemetry)
    assert telemetry.bootstrap_servers == "k:9092"
    assert telemetry.topic == "agent.events"


@pytest.mark.asyncio
async def test_disabled_telemetry_logs_once_and_produces_nothing(
    telemetry_settings: Settings, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_producer(_: str) -> Any:
        raise AssertionError("no producer may be built when telemetry is off")

    monkeypatch.setattr("app.telemetry.aiokafka_producer", no_producer)
    settings = telemetry_settings.model_copy(update={"telemetry_salt": None})
    app = create_app(settings)
    assert app.state.telemetry is None
    app.state.orchestrator = AgentOrchestrator(
        mcp_client=FakeMCP(), llm_router=ScriptedRouter([final_turn()]), skills_loader=app.state.skills_loader
    )

    with caplog.at_level(logging.INFO, logger="sportsagentic"):
        async with app.router.lifespan_context(app):
            response = await send(app, "POST", "/chat", json={"message": PROMPT}, headers=UI_HEADERS)

    assert response.status_code == 200
    assert "X-Request-Id" in response.headers
    off = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Telemetry off")]
    assert off == ["Telemetry off: TELEMETRY_SALT not set, so no agent.events are produced"]


# --- Kafka down and the drop path ---------------------------------------------------------


@pytest.mark.asyncio
async def test_startup_does_not_wait_for_kafka(telemetry_settings: Settings) -> None:
    producer = FakeProducer(hang_on_start=True)
    app = make_app(telemetry_settings, ScriptedRouter([final_turn()]), FakeMCP(), producer)

    began = time.monotonic()
    async with app.router.lifespan_context(app):
        startup_seconds = time.monotonic() - began
        response = await send(app, "POST", "/chat", json={"message": PROMPT}, headers=UI_HEADERS)
    assert startup_seconds < 0.5
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_requests_succeed_and_events_are_dropped_when_kafka_is_down(telemetry_settings: Settings) -> None:
    producer = FakeProducer(start_error=ConnectionError("Unable to bootstrap"))
    app = make_app(telemetry_settings, ScriptedRouter([tool_turn("get_cfb_scores"), final_turn()]), FakeMCP(), producer)
    telemetry = KafkaTelemetry("fake-kafka:9092", producer_factory=lambda _: producer, retry_initial=0.01, retry_max=0.01)
    app.state.telemetry = telemetry

    await telemetry.start()
    response = await send(app, "POST", "/chat", json={"message": PROMPT}, headers=UI_HEADERS)
    for _ in range(50):
        if telemetry.dropped >= 3:
            break
        await asyncio.sleep(0.01)
    await telemetry.stop()

    assert response.status_code == 200
    assert producer.messages == []
    assert telemetry.dropped == 3
    assert telemetry.dropped_by_reason == {DROP_KAFKA_UNAVAILABLE: 3}


@pytest.mark.asyncio
async def test_real_aiokafka_against_a_closed_port_drops_without_blocking(telemetry_settings: Settings) -> None:
    """The real client, pointed at a local port nothing listens on."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    settings = telemetry_settings.model_copy(update={"kafka_bootstrap": f"127.0.0.1:{port}"})
    app = create_app(settings)
    app.state.orchestrator = AgentOrchestrator(
        mcp_client=FakeMCP(), llm_router=ScriptedRouter([final_turn()]), skills_loader=app.state.skills_loader
    )
    telemetry: KafkaTelemetry = app.state.telemetry
    assert isinstance(telemetry, KafkaTelemetry)

    async with app.router.lifespan_context(app):
        began = time.monotonic()
        response = await send(app, "POST", "/chat", json={"message": PROMPT}, headers=UI_HEADERS)
        request_seconds = time.monotonic() - began
        for _ in range(100):
            if telemetry.dropped >= 2:
                break
            await asyncio.sleep(0.02)
    assert response.status_code == 200
    assert request_seconds < 1.0
    assert telemetry.dropped == 2
    assert telemetry.sent == 0


@pytest.mark.asyncio
async def test_full_queue_drops_new_events() -> None:
    telemetry = KafkaTelemetry("k:9092", producer_factory=lambda _: FakeProducer(hang_on_start=True), max_queue=2)
    await telemetry.start()
    for i in range(5):
        telemetry.emit({"request_id": str(i)})
    assert telemetry.dropped_by_reason == {DROP_QUEUE_FULL: 3}
    await telemetry.stop()
    assert telemetry.dropped_by_reason == {DROP_QUEUE_FULL: 3, DROP_KAFKA_UNAVAILABLE: 2}


@pytest.mark.asyncio
async def test_emit_before_start_is_dropped_not_raised() -> None:
    telemetry = KafkaTelemetry("k:9092", producer_factory=lambda _: FakeProducer())
    telemetry.emit({"request_id": "r"})
    assert telemetry.dropped_by_reason == {DROP_KAFKA_UNAVAILABLE: 1}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("producer", "reason"),
    [
        (FakeProducer(send_error=RuntimeError("KafkaTimeoutError")), DROP_SEND_FAILED),
        (FakeProducer(delivery_error=RuntimeError("NotLeaderForPartition")), DROP_DELIVERY_FAILED),
    ],
)
async def test_failed_sends_are_counted(producer: FakeProducer, reason: str) -> None:
    telemetry = KafkaTelemetry("k:9092", producer_factory=lambda _: producer)
    await telemetry.start()
    telemetry.emit({"request_id": "r1"})
    telemetry.emit({"request_id": "r2"})
    for _ in range(50):
        if telemetry.dropped == 2:
            break
        await asyncio.sleep(0.01)
    await telemetry.stop()
    assert telemetry.dropped_by_reason == {reason: 2}
    assert telemetry.sent == 0
    assert producer.stopped


@pytest.mark.asyncio
async def test_stop_flushes_queued_events() -> None:
    producer = FakeProducer()
    telemetry = KafkaTelemetry("k:9092", producer_factory=lambda _: producer)
    await telemetry.start()
    while not telemetry.connected:
        await asyncio.sleep(0)
    for i in range(20):  # queued, and stop() runs before the sender gets a turn
        telemetry.emit({"request_id": f"r{i}", "n": i})
    await telemetry.stop()
    assert [e["n"] for e in producer.events()] == list(range(20))
    assert telemetry.sent == 20 and telemetry.dropped == 0
    assert [m["key"] for m in producer.messages][:2] == [b"r0", b"r1"]


def test_drops_are_logged_at_most_once_a_minute(caplog: pytest.LogCaptureFixture) -> None:
    now = [1000.0]
    telemetry = KafkaTelemetry("k:9092", clock=lambda: now[0])

    with caplog.at_level(logging.WARNING, logger="sportsagentic.telemetry"):
        for _ in range(5):
            telemetry.emit({"request_id": "r"})
        now[0] += 30
        telemetry.emit({"request_id": "r"})
        now[0] += 31
        telemetry.emit({"request_id": "r"})

    warnings = [r.getMessage() for r in caplog.records if "Dropped" in r.getMessage()]
    assert len(warnings) == 2
    assert warnings[0].startswith("Dropped 1 telemetry event(s)")
    assert warnings[1].startswith("Dropped 6 telemetry event(s)")
    assert "7 in total" in warnings[1]


@pytest.mark.asyncio
async def test_reporter_logs_remaining_drops_after_the_interval(caplog: pytest.LogCaptureFixture) -> None:
    telemetry = KafkaTelemetry(
        "k:9092", producer_factory=lambda _: FakeProducer(hang_on_start=True), max_queue=1, drop_log_interval=0.05
    )
    await telemetry.start()
    with caplog.at_level(logging.WARNING, logger="sportsagentic.telemetry"):
        for _ in range(4):
            telemetry.emit({"request_id": "r"})
        await asyncio.sleep(0.12)
    await telemetry.stop()
    warnings = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Dropped")]
    assert warnings[0].startswith("Dropped 1 ")
    assert warnings[1].startswith("Dropped 2 ")


# --- token usage parsing ------------------------------------------------------------------


def test_usage_is_read_from_a_completion() -> None:
    response = parse_openai_chat_response(
        {
            "choices": [{"message": {"role": "assistant", "content": "Hi"}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
        },
        provider_name="gemini",
    )
    assert (response.input_tokens, response.output_tokens) == (12, 3)


@pytest.mark.parametrize("usage", [None, {}, {"prompt_tokens": "12"}, {"prompt_tokens": -1}, {"prompt_tokens": True}])
def test_missing_or_bad_usage_is_none(usage: Any) -> None:
    payload: Dict[str, Any] = {"choices": [{"message": {"role": "assistant", "content": "Hi"}}]}
    if usage is not None:
        payload["usage"] = usage
    response = parse_openai_chat_response(payload, provider_name="llama")
    assert response.input_tokens is None and response.output_tokens is None


def test_usage_is_read_from_the_last_stream_chunk() -> None:
    assembler = ChatStreamAssembler("llama", lambda _: None)
    assembler.feed({"choices": [{"delta": {"content": "Hi"}}]})
    assembler.feed({"choices": [{"delta": {}, "finish_reason": "stop"}]})
    assembler.feed({"choices": [], "usage": {"prompt_tokens": 40, "completion_tokens": 2}})
    response = assembler.finish()
    assert response.content == "Hi"
    assert (response.input_tokens, response.output_tokens) == (40, 2)


def test_event_time_is_rfc3339_utc() -> None:
    from datetime import timedelta, timezone

    from app.telemetry.events import rfc3339_utc

    moment = datetime(2026, 11, 11, 13, 5, 40, 123456, tzinfo=timezone(timedelta(hours=-7)))
    assert rfc3339_utc(moment) == "2026-11-11T20:05:40.123Z"


def test_key_partitioner_matches_kafka_default() -> None:
    from aiokafka.partitioner import DefaultPartitioner

    from app.telemetry.producer import KeyPartitioner

    partitioner = KeyPartitioner(DefaultPartitioner())
    keys = [str(uuid.uuid4()).encode() for _ in range(50)]
    assert {partitioner(k, [0], [0]) for k in keys} == {0}
    for key in keys:
        assert partitioner(key, [0, 1, 2], [0, 1, 2]) == DefaultPartitioner()(key, [0, 1, 2], [0, 1, 2])
