"""Unit tests for LlamaServerClient, GeminiClient, and LLMRouter failover."""

from __future__ import annotations

import logging
import httpx
import pytest

from app.llm.base import LLMClientError
from app.llm.gemini_client import GeminiClient
from app.llm.llama_client import LlamaServerClient
from app.llm.router import LLMRouter
from app.mcp.client import DEFAULT_CFB_TOOLS


def _openai_completion_payload(content: str = "", tool_calls: list | None = None) -> dict:
    msg: dict = {"role": "assistant", "content": content}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    return {"choices": [{"index": 0, "message": msg}]}


@pytest.mark.asyncio
async def test_router_primary_llama_success() -> None:
    """When llama-server succeeds in auto mode, Gemini is not called."""
    gemini_called = False

    def llama_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(200, json=_openai_completion_payload("Roll Tide! Bama won 42-10."))

    def gemini_handler(request: httpx.Request) -> httpx.Response:
        nonlocal gemini_called
        gemini_called = True
        return httpx.Response(200, json=_openai_completion_payload("From Gemini"))

    async with httpx.AsyncClient(transport=httpx.MockTransport(llama_handler)) as llama_http, \
               httpx.AsyncClient(transport=httpx.MockTransport(gemini_handler)) as gemini_http:
        llama = LlamaServerClient(http_client=llama_http)
        gemini = GeminiClient(api_key="test-key", http_client=gemini_http)
        router = LLMRouter(llama_client=llama, gemini_client=gemini, default_provider="auto")

        resp = await router.chat([{"role": "user", "content": "Did Bama win?"}])
        assert resp.content == "Roll Tide! Bama won 42-10."
        assert resp.provider_used == "llama"
        assert resp.fallback_triggered is False
        assert gemini_called is False

        ok, _ = await llama.check_health()
        assert ok is True


@pytest.mark.asyncio
async def test_router_failover_to_gemini_on_llama_timeout(caplog: pytest.LogCaptureFixture) -> None:
    """When llama-server times out, LLMRouter logs a warning and fails over to Gemini."""

    def llama_timeout_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("llama-server read timed out")

    def gemini_handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("Authorization") == "Bearer test-gemini-key"
        return httpx.Response(
            200,
            json=_openai_completion_payload("Gemini backup answer: Texas beat Oklahoma 34-3."),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(llama_timeout_handler)) as llama_http, \
               httpx.AsyncClient(transport=httpx.MockTransport(gemini_handler)) as gemini_http:
        llama = LlamaServerClient(timeout_seconds=1.5, http_client=llama_http)
        gemini = GeminiClient(api_key="test-gemini-key", http_client=gemini_http)
        router = LLMRouter(llama_client=llama, gemini_client=gemini, default_provider="auto")

        with caplog.at_level(logging.WARNING, logger="sportsagentic"):
            resp = await router.chat(
                [{"role": "user", "content": "Red River Rivalry score?"}],
                mcp_tools=DEFAULT_CFB_TOOLS,
            )

        assert resp.content == "Gemini backup answer: Texas beat Oklahoma 34-3."
        assert resp.provider_used == "gemini"
        assert resp.fallback_triggered is True
        assert "failing over to Gemini backup" in caplog.text


@pytest.mark.asyncio
async def test_router_failover_to_gemini_When_llama_is_down() -> None:
    """When llama-server refuses connection or returns 503, LLMRouter fails over to Gemini."""

    def llama_down_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="Model loading / server busy")

    def gemini_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_openai_completion_payload(
                tool_calls=[
                    {
                        "id": "call_gem_1",
                        "type": "function",
                        "function": {
                            "name": "get_cfb_rankings",
                            "arguments": '{"week": 8}',
                        },
                    }
                ]
            ),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(llama_down_handler)) as llama_http, \
               httpx.AsyncClient(transport=httpx.MockTransport(gemini_handler)) as gemini_http:
        llama = LlamaServerClient(http_client=llama_http)
        gemini = GeminiClient(api_key="test-gemini-key", http_client=gemini_http)
        router = LLMRouter(llama_client=llama, gemini_client=gemini, default_provider="auto")

        resp = await router.chat(
            [{"role": "user", "content": "Week 8 rankings"}],
            mcp_tools=DEFAULT_CFB_TOOLS,
        )
        assert resp.provider_used == "gemini"
        assert resp.fallback_triggered is True
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "get_cfb_rankings"
        assert resp.tool_calls[0].arguments == {"week": 8}


@pytest.mark.asyncio
async def test_router_explicit_provider_selection() -> None:
    """Explicit provider='gemini' skips llama-server; provider='llama' does not fail over."""
    llama_called = False

    def llama_handler(request: httpx.Request) -> httpx.Response:
        nonlocal llama_called
        llama_called = True
        raise httpx.ConnectError("Connection refused")

    def gemini_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_openai_completion_payload("Direct Gemini reply"))

    async with httpx.AsyncClient(transport=httpx.MockTransport(llama_handler)) as llama_http, \
               httpx.AsyncClient(transport=httpx.MockTransport(gemini_handler)) as gemini_http:
        llama = LlamaServerClient(http_client=llama_http)
        gemini = GeminiClient(api_key="test-gemini-key", http_client=gemini_http)
        router = LLMRouter(llama_client=llama, gemini_client=gemini, default_provider="auto")

        # Direct Gemini choice
        direct_gem = await router.chat(
            [{"role": "user", "content": "Hello"}],
            provider="gemini",
        )
        assert direct_gem.content == "Direct Gemini reply"
        assert direct_gem.provider_used == "gemini"
        assert direct_gem.fallback_triggered is False
        assert llama_called is False

        # Direct llama choice raises when llama is down (no failover)
        with pytest.raises(LLMClientError, match="Could not connect to llama-server"):
            await router.chat([{"role": "user", "content": "Hello"}], provider="llama")


@pytest.mark.asyncio
async def test_router_raises_when_both_llama_and_gemini_fail() -> None:
    """If llama-server fails and Gemini API key is missing, LLMRouter raises combined error."""

    def llama_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused")

    async with httpx.AsyncClient(transport=httpx.MockTransport(llama_handler)) as llama_http:
        llama = LlamaServerClient(http_client=llama_http)
        gemini = GeminiClient(api_key=None)
        router = LLMRouter(llama_client=llama, gemini_client=gemini)

        with pytest.raises(LLMClientError, match="Gemini backup also failed"):
            await router.chat([{"role": "user", "content": "Scores?"}])


@pytest.mark.asyncio
async def test_inline_json_tool_call_fallback_parsing() -> None:
    """Verify local models that emit JSON tool calls in content are parsed into ToolCallRequests."""

    def llama_inline_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_openai_completion_payload(
                content='```json\n{"name": "get_team_schedule", "arguments": {"team": "CSU"}}\n```'
            ),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(llama_inline_handler)) as llama_http:
        llama = LlamaServerClient(http_client=llama_http)
        resp = await llama.chat(
            [{"role": "user", "content": "CSU schedule"}],
            mcp_tools=DEFAULT_CFB_TOOLS,
        )
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "get_team_schedule"
        assert resp.tool_calls[0].arguments == {"team": "CSU"}
        assert resp.content == ""
