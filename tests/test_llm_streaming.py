"""Tests for token streaming through the LLM clients, router, and /chat/stream."""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import httpx
import pytest

from app.agent.orchestrator import AgentOrchestrator
from app.agent.streaming import UNAVAILABLE_MESSAGE, stream_agent_turn
from app.llm.base import LLMClientError, LLMConnectionError, LLMTimeoutError, parse_openai_chat_response
from app.llm.gemini_client import GeminiClient
from app.llm.llama_client import LlamaServerClient
from app.llm.router import LLMRouter
from app.llm.streaming import ChatStreamAssembler
from app.mcp.client import DEFAULT_CFB_TOOLS
from app.skills.loader import SkillsLoader

TOOL_NAMES = [tool["name"] for tool in DEFAULT_CFB_TOOLS]
INLINE_CALL = '{"name": "get_team_schedule", "arguments": {"team": "CSU"}}'
USER_MESSAGES = [{"role": "user", "content": "Scores?"}]


def sse(*chunks: Dict[str, Any], done: bool = True) -> bytes:
    frames = [f"data: {json.dumps(chunk)}\n\n" for chunk in chunks]
    if done:
        frames.append("data: [DONE]\n\n")
    return "".join(frames).encode()


def text_chunk(text: str) -> Dict[str, Any]:
    return {"choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]}


def tool_chunk(*tool_calls: Dict[str, Any]) -> Dict[str, Any]:
    return {"choices": [{"index": 0, "delta": {"tool_calls": list(tool_calls)}, "finish_reason": None}]}


def stop_chunk() -> Dict[str, Any]:
    return {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}


def text_stream(*texts: str) -> bytes:
    return sse(*[text_chunk(text) for text in texts], stop_chunk())


def sse_response(body: bytes) -> httpx.Response:
    return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})


def failing_response(body: bytes) -> httpx.Response:
    """A stream that delivers body and then loses the connection."""

    async def chunks() -> AsyncIterator[bytes]:
        yield body
        raise httpx.ReadError("connection lost")

    return httpx.Response(200, content=chunks(), headers={"content-type": "text/event-stream"})


def feed_text(pieces: List[str], tool_names: Optional[List[str]] = None) -> Tuple[List[str], Any]:
    sent: List[str] = []
    assembler = ChatStreamAssembler("llama", sent.append, TOOL_NAMES if tool_names is None else tool_names)
    for piece in pieces:
        assembler.feed(text_chunk(piece))
    return sent, assembler.finish()


def make_router(
    llama_handler: Any,
    gemini_handler: Any,
    default_provider: str = "auto",
) -> LLMRouter:
    return LLMRouter(
        llama_client=LlamaServerClient(
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(llama_handler))
        ),
        gemini_client=GeminiClient(
            api_key="test-key",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(gemini_handler)),
        ),
        default_provider=default_provider,  # type: ignore[arg-type]
    )


def fail_if_called(request: httpx.Request) -> httpx.Response:
    raise AssertionError("provider should not be called")


class StubMCP:
    def __init__(self) -> None:
        self.calls: List[Tuple[str, Dict[str, Any]]] = []

    async def list_tools(self, fallback_to_defaults: bool = True) -> List[Dict[str, Any]]:
        return DEFAULT_CFB_TOOLS

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        self.calls.append((name, arguments))
        return "[]"


def make_agent(router: LLMRouter, sample_skills_dir: Any) -> AgentOrchestrator:
    return AgentOrchestrator(
        mcp_client=StubMCP(),  # type: ignore[arg-type]
        llm_router=router,
        skills_loader=SkillsLoader(skills_dir=str(sample_skills_dir)),
        max_tool_iterations=3,
    )


async def collect_frames(agent: AgentOrchestrator) -> List[Tuple[str, Dict[str, Any]]]:
    frames = []
    async for frame in stream_agent_turn(agent, user_message="Scores?"):
        if frame.startswith(":"):
            continue
        event, data = frame.strip().split("\n")
        frames.append((event.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return frames


def test_text_is_passed_on_as_it_arrives() -> None:
    sent, response = feed_text(["Georgia ", "is ", "#1."])
    assert sent == ["Georgia ", "is ", "#1."]
    assert response.content == "Georgia is #1."
    assert response.tool_calls == []


def test_tool_call_is_assembled_from_fragments() -> None:
    assembler = ChatStreamAssembler("llama", lambda text: None, TOOL_NAMES)
    assembler.feed(
        tool_chunk(
            {
                "index": 0,
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_cfb_scores", "arguments": ""},
            }
        )
    )
    assembler.feed(tool_chunk({"index": 0, "function": {"arguments": '{"team": "Ole'}}))
    assembler.feed(tool_chunk({"index": 0, "function": {"arguments": ' Miss", "week": 9}'}}))

    response = assembler.finish()
    assert [(c.id, c.name, c.arguments) for c in response.tool_calls] == [
        ("call_1", "get_cfb_scores", {"team": "Ole Miss", "week": 9}),
    ]


def test_parallel_tool_calls_keep_their_own_arguments() -> None:
    assembler = ChatStreamAssembler("llama", lambda text: None, TOOL_NAMES)
    assembler.feed(
        tool_chunk(
            {"index": 0, "id": "a", "function": {"name": "get_cfb_scores", "arguments": '{"we'}},
            {"index": 1, "id": "b", "function": {"name": "get_cfb_rankings", "arguments": '{"ye'}},
        )
    )
    assembler.feed(
        tool_chunk(
            {"index": 1, "function": {"arguments": 'ar": 2025}'}},
            {"index": 0, "function": {"arguments": 'ek": 9}'}},
        )
    )

    response = assembler.finish()
    assert [(c.id, c.name, c.arguments) for c in response.tool_calls] == [
        ("a", "get_cfb_scores", {"week": 9}),
        ("b", "get_cfb_rankings", {"year": 2025}),
    ]


def test_new_id_at_the_same_index_starts_a_new_call() -> None:
    assembler = ChatStreamAssembler("gemini", lambda text: None, TOOL_NAMES)
    assembler.feed(tool_chunk({"index": 0, "id": "a", "function": {"name": "get_cfb_scores", "arguments": "{}"}}))
    assembler.feed(tool_chunk({"index": 0, "id": "b", "function": {"name": "get_cfb_rankings", "arguments": "{}"}}))

    assert [c.name for c in assembler.finish().tool_calls] == ["get_cfb_scores", "get_cfb_rankings"]


def test_tool_call_without_an_index_is_accepted() -> None:
    assembler = ChatStreamAssembler("gemini", lambda text: None, TOOL_NAMES)
    assembler.feed(tool_chunk({"id": "a", "function": {"name": "get_cfb_scores", "arguments": '{"week"'}}))
    assembler.feed(tool_chunk({"function": {"arguments": ": 9}"}}))

    assert assembler.finish().tool_calls[0].arguments == {"week": 9}


def test_text_before_a_native_tool_call_stays_in_the_response() -> None:
    sent: List[str] = []
    assembler = ChatStreamAssembler("llama", sent.append, TOOL_NAMES)
    assembler.feed(text_chunk("Checking."))
    assembler.feed(tool_chunk({"index": 0, "id": "a", "function": {"name": "get_cfb_scores", "arguments": "{}"}}))
    assembler.feed(text_chunk(" Late text."))

    response = assembler.finish()
    assert sent == ["Checking."]
    assert response.content == "Checking. Late text."
    assert len(response.tool_calls) == 1


@pytest.mark.parametrize("split", [1, 3, 7, 1000])
@pytest.mark.parametrize(
    "content",
    [
        INLINE_CALL,
        f"  \n{INLINE_CALL}\n",
        f"```json\n{INLINE_CALL}\n```",
        f"```\n{INLINE_CALL}\n```",
        f"Let me check.\n```json\n{INLINE_CALL}\n```",
        f"Let me check.\n```json\n{INLINE_CALL}\n```\nOne moment.",
        f"````json\n{INLINE_CALL}\n```",
        f"```python\nprint(1)\n```\n```json\n{INLINE_CALL}\n```",
    ],
)
def test_inline_tool_calls_are_never_shown(content: str, split: int) -> None:
    pieces = [content[i : i + split] for i in range(0, len(content), split)]
    sent, response = feed_text(pieces)

    assert "get_team_schedule" not in "".join(sent)
    assert "```json" not in "".join(sent)
    assert [c.name for c in response.tool_calls] == ["get_team_schedule"]
    assert response.tool_calls[0].arguments == {"team": "CSU"}
    assert response.content == ""


@pytest.mark.parametrize("split", [1, 3, 7, 1000])
@pytest.mark.parametrize(
    "content",
    [
        "Georgia is #1.",
        "   ",
        "{not json at all}",
        '{"name": "not_a_tool", "arguments": {}}',
        f"{INLINE_CALL} and then some prose",
        f"Prose first {INLINE_CALL}",
        "Use `code` and ``double`` ticks.",
        "```python\nprint(1)\n```\nThat is the code.",
        '```json\n{"team": "CSU"}\n```\nThat is the data.',
        f'```json\n{{"team": "CSU"}}\n```\n```json\n{INLINE_CALL}\n```',
        "```json\n[1, 2]\n```",
        '```json\n{"unterminated": true',
        "Ends with a tick`",
        "Ends with an open fence ```",
    ],
)
def test_ordinary_text_is_sent_whole_and_matches_chat(content: str, split: int) -> None:
    pieces = [content[i : i + split] for i in range(0, len(content), split)]
    sent, response = feed_text(pieces)

    expected = parse_openai_chat_response(
        {"choices": [{"message": {"content": content}}]},
        provider_name="llama",
        known_tool_names=TOOL_NAMES,
    )
    assert expected.tool_calls == []
    assert "".join(sent) == content
    assert response.content == expected.content
    assert response.tool_calls == []


def test_prose_streams_before_a_fence_is_held() -> None:
    sent, _ = feed_text(["Let me ", "check.\n", "``", "`json\n", INLINE_CALL, "\n```"])
    assert sent == ["Let me ", "check.\n"]


def test_ordinary_code_block_keeps_streaming() -> None:
    sent, _ = feed_text(["```python\n", "print(1)\n", "```\n", "Done."])
    assert sent[:2] == ["```python\n", "print(1)\n"]


def test_json_reply_without_tools_is_not_held() -> None:
    sent, _ = feed_text(['{"a"', ": 1}"], tool_names=[])
    assert sent == ['{"a"', ": 1}"]


@pytest.mark.asyncio
async def test_llama_requests_a_stream_and_reports_deltas() -> None:
    captured: List[Dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return sse_response(text_stream("Roll ", "Tide"))

    sent: List[str] = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        response = await LlamaServerClient(http_client=http).chat_stream(
            USER_MESSAGES, sent.append, mcp_tools=DEFAULT_CFB_TOOLS
        )

    assert captured[0]["stream"] is True
    assert captured[0]["tools"]
    assert sent == ["Roll ", "Tide"]
    assert response.content == "Roll Tide"
    assert response.provider_used == "llama"


@pytest.mark.asyncio
async def test_gemini_stream_preserves_thought_signatures(sample_skills_dir: Any) -> None:
    captured: List[Dict[str, Any]] = []
    signature = {"google": {"thought_signature": "signature_abc"}}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer test-key"
        captured.append(json.loads(request.content))
        if len(captured) == 2:
            return sse_response(text_stream("Done"))
        return sse_response(
            sse(
                tool_chunk(
                    {
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_cfb_scores", "arguments": '{"team": "Ole Miss"}'},
                        "extra_content": signature,
                    }
                ),
                stop_chunk(),
            )
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        router = LLMRouter(
            llama_client=LlamaServerClient(),
            gemini_client=GeminiClient(api_key="test-key", http_client=http),
            default_provider="gemini",
        )
        await make_agent(router, sample_skills_dir).run("Ole Miss", on_text_delta=lambda text: None)

    assert captured[1]["stream"] is True
    replayed_call = captured[1]["messages"][-2]["tool_calls"][0]
    assert replayed_call["extra_content"] == signature


@pytest.mark.asyncio
async def test_gemini_stream_requires_an_api_key() -> None:
    with pytest.raises(LLMClientError, match="not configured"):
        await GeminiClient(api_key=None).chat_stream(USER_MESSAGES, lambda text: None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response,error,match",
    [
        (httpx.Response(503, text="busy"), LLMConnectionError, "HTTP 503"),
        (httpx.Response(400, text="bad tools"), LLMClientError, "bad tools"),
        (sse_response(sse(text_chunk("Partial"), done=False)), LLMConnectionError, "before the reply"),
        (sse_response(b'error: {"code": 500, "message": "boom"}\n\n'), LLMConnectionError, "boom"),
        (sse_response(b'data: {"error": {"message": "boom"}}\n\n'), LLMConnectionError, "boom"),
        (sse_response(b"data: not-json\n\n"), LLMClientError, "Invalid JSON"),
    ],
)
async def test_llama_stream_failures_map_to_client_errors(
    response: httpx.Response, error: type, match: str
) -> None:
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: response)) as http:
        with pytest.raises(error, match=match):
            await LlamaServerClient(http_client=http).chat_stream(USER_MESSAGES, lambda text: None)


@pytest.mark.asyncio
async def test_llama_stream_timeout_maps_to_timeout_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(LLMTimeoutError):
            await LlamaServerClient(http_client=http).chat_stream(USER_MESSAGES, lambda text: None)


@pytest.mark.asyncio
async def test_finish_reason_alone_completes_a_stream() -> None:
    body = sse(text_chunk("Hi"), stop_chunk(), done=False)
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: sse_response(body))) as http:
        response = await LlamaServerClient(http_client=http).chat_stream(USER_MESSAGES, lambda text: None)
    assert response.content == "Hi"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "llama_handler",
    [
        lambda request: httpx.Response(503, text="busy"),
        lambda request: failing_response(b""),
        # Nothing from these partial replies has reached the user yet.
        lambda request: failing_response(sse(text_chunk('{"name": "get_team'), done=False)),
        lambda request: failing_response(
            sse(tool_chunk({"index": 0, "id": "a", "function": {"name": "get_cfb_scores"}}), done=False)
        ),
    ],
)
async def test_auto_fails_over_before_any_text_is_sent(llama_handler: Any) -> None:
    router = make_router(llama_handler, lambda request: sse_response(text_stream("From ", "Gemini")))
    sent: List[str] = []

    response = await router.chat_stream(USER_MESSAGES, sent.append, mcp_tools=DEFAULT_CFB_TOOLS)

    assert sent == ["From ", "Gemini"]
    assert response.provider_used == "gemini"
    assert response.fallback_triggered is True


@pytest.mark.asyncio
async def test_auto_does_not_fail_over_after_text_is_sent() -> None:
    router = make_router(
        lambda request: failing_response(sse(text_chunk("Georgia "), done=False)),
        fail_if_called,
    )
    sent: List[str] = []

    with pytest.raises(LLMConnectionError):
        await router.chat_stream(USER_MESSAGES, sent.append, mcp_tools=DEFAULT_CFB_TOOLS)

    assert sent == ["Georgia "]


@pytest.mark.asyncio
async def test_auto_does_not_fail_over_when_the_caller_forbids_it() -> None:
    router = make_router(lambda request: httpx.Response(503, text="busy"), fail_if_called)

    with pytest.raises(LLMConnectionError):
        await router.chat_stream(USER_MESSAGES, lambda text: None, allow_fallback=False)


@pytest.mark.asyncio
async def test_auto_raises_a_combined_error_when_both_providers_fail() -> None:
    router = make_router(
        lambda request: httpx.Response(503, text="busy"),
        lambda request: httpx.Response(500, text="down"),
    )

    with pytest.raises(LLMClientError, match="Gemini backup also failed"):
        await router.chat_stream(USER_MESSAGES, lambda text: None)


@pytest.mark.asyncio
async def test_explicit_providers_do_not_fail_over() -> None:
    router = make_router(lambda request: httpx.Response(503, text="busy"), fail_if_called)
    with pytest.raises(LLMConnectionError):
        await router.chat_stream(USER_MESSAGES, lambda text: None, provider="llama")

    router = make_router(fail_if_called, lambda request: sse_response(text_stream("Hi")))
    response = await router.chat_stream(USER_MESSAGES, lambda text: None, provider="gemini")
    assert response.provider_used == "gemini"
    assert response.fallback_triggered is False


@pytest.mark.asyncio
async def test_stream_emits_one_delta_per_token(sample_skills_dir: Any) -> None:
    router = make_router(lambda request: sse_response(text_stream("Georgia ", "is ", "#1.")), fail_if_called)

    assert await collect_frames(make_agent(router, sample_skills_dir)) == [
        ("message.delta", {"text": "Georgia "}),
        ("message.delta", {"text": "is "}),
        ("message.delta", {"text": "#1."}),
        ("done", {}),
    ]


@pytest.mark.asyncio
async def test_stream_runs_inline_tool_call_without_showing_it(sample_skills_dir: Any) -> None:
    replies = iter(
        [
            text_stream(INLINE_CALL[:20], INLINE_CALL[20:]),
            text_stream("CSU plays ", "Saturday."),
        ]
    )
    router = make_router(lambda request: sse_response(next(replies)), fail_if_called)
    agent = make_agent(router, sample_skills_dir)

    assert await collect_frames(agent) == [
        (
            "tool.start",
            {"id": "1:call_inline_1", "name": "get_team_schedule", "args": {"team": "CSU"}},
        ),
        ("tool.end", {"id": "1:call_inline_1", "name": "get_team_schedule", "ok": True}),
        ("message.delta", {"text": "CSU plays "}),
        ("message.delta", {"text": "Saturday."}),
        ("done", {}),
    ]


@pytest.mark.asyncio
async def test_stream_separates_text_from_different_iterations(sample_skills_dir: Any) -> None:
    replies = iter(
        [
            sse(
                text_chunk("Checking."),
                tool_chunk({"index": 0, "id": "a", "function": {"name": "get_cfb_scores", "arguments": "{}"}}),
                stop_chunk(),
            ),
            text_stream("Georgia ", "won."),
        ]
    )
    router = make_router(lambda request: sse_response(next(replies)), fail_if_called)

    frames = await collect_frames(make_agent(router, sample_skills_dir))

    assert "".join(data["text"] for event, data in frames if event == "message.delta") == (
        "Checking.\n\nGeorgia won."
    )


@pytest.mark.asyncio
async def test_stream_fails_over_silently_before_text(sample_skills_dir: Any) -> None:
    router = make_router(
        lambda request: httpx.Response(503, text="busy"),
        lambda request: sse_response(text_stream("From ", "Gemini")),
    )

    assert await collect_frames(make_agent(router, sample_skills_dir)) == [
        ("message.delta", {"text": "From "}),
        ("message.delta", {"text": "Gemini"}),
        ("done", {}),
    ]


@pytest.mark.asyncio
async def test_stream_sends_generic_error_after_text(sample_skills_dir: Any) -> None:
    router = make_router(
        lambda request: failing_response(sse(text_chunk("Georgia "), done=False)),
        fail_if_called,
    )

    assert await collect_frames(make_agent(router, sample_skills_dir)) == [
        ("message.delta", {"text": "Georgia "}),
        ("error", {"message": UNAVAILABLE_MESSAGE}),
    ]


@pytest.mark.asyncio
async def test_stream_does_not_fail_over_after_text_from_an_earlier_iteration(
    sample_skills_dir: Any,
) -> None:
    replies = iter(
        [
            sse_response(
                sse(
                    text_chunk("Checking."),
                    tool_chunk({"index": 0, "id": "a", "function": {"name": "get_cfb_scores", "arguments": "{}"}}),
                    stop_chunk(),
                )
            ),
            httpx.Response(503, text="busy"),
        ]
    )
    router = make_router(lambda request: next(replies), fail_if_called)

    assert await collect_frames(make_agent(router, sample_skills_dir)) == [
        ("message.delta", {"text": "Checking."}),
        ("tool.start", {"id": "1:a", "name": "get_cfb_scores", "args": {}}),
        ("tool.end", {"id": "1:a", "name": "get_cfb_scores", "ok": True}),
        ("error", {"message": UNAVAILABLE_MESSAGE}),
    ]


@pytest.mark.asyncio
async def test_stream_error_text_hides_provider_details(sample_skills_dir: Any) -> None:
    router = make_router(
        lambda request: sse_response(sse(text_chunk("Hi"), done=False) + b'error: {"message": "secret"}\n\n'),
        fail_if_called,
    )

    frames = await collect_frames(make_agent(router, sample_skills_dir))

    assert frames[-1] == ("error", {"message": UNAVAILABLE_MESSAGE})


@pytest.mark.asyncio
async def test_disconnect_cancels_a_streaming_provider_call(sample_skills_dir: Any) -> None:
    cancelled = asyncio.Event()

    async def hanging_body() -> AsyncIterator[bytes]:
        yield sse(text_chunk("Hi"), done=False)
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    router = make_router(lambda request: httpx.Response(200, content=hanging_body()), fail_if_called)
    stream = stream_agent_turn(make_agent(router, sample_skills_dir), user_message="Hi")

    assert (await stream.__anext__()).startswith("event: message.delta")
    await stream.aclose()
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_run_without_a_text_callback_stays_non_streaming(sample_skills_dir: Any) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "stream" not in json.loads(request.content)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "Plain reply"}}]},
        )

    response = await make_agent(make_router(handler, fail_if_called), sample_skills_dir).run("Hi")

    assert response.reply == "Plain reply"
