"""Unit tests for the AgentOrchestrator multi-step tool-calling loop."""

from __future__ import annotations

import json
from pathlib import Path
import httpx
import pytest

from app.agent.orchestrator import AgentOrchestrator
from app.llm.gemini_client import GeminiClient
from app.llm.llama_client import LlamaServerClient
from app.llm.router import LLMRouter
from app.mcp.client import DEFAULT_CFB_TOOLS, SportsAgenticMCPClient
from app.skills.loader import SkillsLoader


@pytest.mark.asyncio
async def test_orchestrator_executes_mcp_tool_and_synthesizes_answer(
    sample_skills_dir: Path,
) -> None:
    """Verify full loop: user query -> LLM requests tool -> MCP executes tool -> LLM synthesizes answer."""
    llm_turn = 0

    def mcp_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/sse":
            return httpx.Response(200, content=b"data: /message?sessionId=orch-1\n\n")
        body = json.loads(request.content.decode("utf-8"))
        if body["method"] == "tools/list":
            return httpx.Response(
                200,
                json={"jsonrpc": "2.0", "id": body["id"], "result": {"tools": DEFAULT_CFB_TOOLS}},
            )
        if body["method"] == "tools/call":
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": body["id"],
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": '[{"home_team": "Alabama", "home_points": 31, "away_team": "Georgia", "away_points": 24}]',
                            }
                        ]
                    },
                },
            )
        return httpx.Response(404)

    def llama_handler(request: httpx.Request) -> httpx.Response:
        nonlocal llm_turn
        llm_turn += 1
        if llm_turn == 1:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [
                                    {
                                        "id": "call_bama_1",
                                        "type": "function",
                                        "function": {
                                            "name": "get_cfb_scores",
                                            "arguments": '{"team": "Alabama"}',
                                        },
                                    }
                                ],
                            },
                        }
                    ]
                },
            )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Alabama defeated Georgia 31-24.",
                        },
                    }
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(mcp_handler)) as mcp_http, \
               httpx.AsyncClient(transport=httpx.MockTransport(llama_handler)) as llama_http:
        mcp_client = SportsAgenticMCPClient(http_client=mcp_http)
        llama_client = LlamaServerClient(http_client=llama_http)
        gemini_client = GeminiClient(api_key="test-key")
        router = LLMRouter(llama_client=llama_client, gemini_client=gemini_client)
        skills_loader = SkillsLoader(skills_dir=sample_skills_dir, extra_dirs=[])

        orchestrator = AgentOrchestrator(
            mcp_client=mcp_client,
            llm_router=router,
            skills_loader=skills_loader,
            max_tool_iterations=3,
        )

        result = await orchestrator.run("What was the Alabama score?")
        assert result.reply == "Alabama defeated Georgia 31-24."
        assert result.provider_used == "llama"
        assert result.fallback_triggered is False
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "get_cfb_scores"
        assert result.tool_calls[0].arguments == {"team": "Alabama"}


@pytest.mark.asyncio
async def test_orchestrator_mid_turn_failover_and_max_iterations(
    sample_skills_dir: Path,
) -> None:
    """Verify orchestrator handles max_tool_iterations and fails over to Gemini if llama-server times out."""

    def mcp_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/sse":
            return httpx.Response(200, content=b"data: /message?sessionId=orch-2\n\n")
        body = json.loads(request.content.decode("utf-8"))
        if body["method"] == "tools/list":
            return httpx.Response(
                200,
                json={"jsonrpc": "2.0", "id": body["id"], "result": {"tools": DEFAULT_CFB_TOOLS}},
            )
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": body["id"],
                "result": {"content": [{"type": "text", "text": "[]"}]},
            },
        )

    def llama_handler(request: httpx.Request) -> httpx.Response:
        # Always times out -> forces Gemini failover
        raise httpx.ReadTimeout("llama-server timeout")

    gemini_calls = 0

    def gemini_handler(request: httpx.Request) -> httpx.Response:
        nonlocal gemini_calls
        gemini_calls += 1
        payload = json.loads(request.content.decode("utf-8"))
        # If tools are provided, request a tool call; when max_tool_iterations forces tools=None, synthesize reply
        if "tools" in payload:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [
                                    {
                                        "id": f"call_{gemini_calls}",
                                        "type": "function",
                                        "function": {
                                            "name": "get_cfb_rankings",
                                            "arguments": '{"week": 1}',
                                        },
                                    }
                                ],
                            },
                        }
                    ]
                },
            )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Synthesized answer after reaching max tool iterations via Gemini.",
                        },
                    }
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(mcp_handler)) as mcp_http, \
               httpx.AsyncClient(transport=httpx.MockTransport(llama_handler)) as llama_http, \
               httpx.AsyncClient(transport=httpx.MockTransport(gemini_handler)) as gemini_http:
        mcp_client = SportsAgenticMCPClient(http_client=mcp_http)
        llama_client = LlamaServerClient(http_client=llama_http)
        gemini_client = GeminiClient(api_key="test-key", http_client=gemini_http)
        router = LLMRouter(llama_client=llama_client, gemini_client=gemini_client)
        skills_loader = SkillsLoader(skills_dir=sample_skills_dir, extra_dirs=[])

        orchestrator = AgentOrchestrator(
            mcp_client=mcp_client,
            llm_router=router,
            skills_loader=skills_loader,
            max_tool_iterations=2,
        )

        result = await orchestrator.run("Check week 1 rankings")
        assert "Synthesized answer" in result.reply
        assert result.provider_used == "gemini"
        assert result.fallback_triggered is True
        assert len(result.tool_calls) == 2
