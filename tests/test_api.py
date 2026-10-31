"""Integration tests for the FastAPI endpoints (/chat, /health, /tools, /skills)."""

from __future__ import annotations

import json
import httpx
import pytest

from app.agent.orchestrator import AgentOrchestrator
from app.config import Settings
from app.llm.gemini_client import GeminiClient
from app.llm.llama_client import LlamaServerClient
from app.llm.router import LLMRouter
from app.main import create_app
from app.mcp.client import DEFAULT_CFB_TOOLS, SportsAgenticMCPClient


@pytest.mark.asyncio
async def test_api_endpoints_end_to_end(test_settings: Settings) -> None:
    """Verify /health, /tools, /skills, and /chat endpoints work as expected."""
    app = create_app(test_settings)

    def mcp_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/sse":
            return httpx.Response(200, content=b"data: /message?sessionId=api-1\n\n")
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
                "result": {
                    "content": [{"type": "text", "text": '[{"team": "Ole Miss", "wins": 10}]'}]
                },
            },
        )

    def llama_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Ole Miss has 10 wins on the season.",
                        },
                    }
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(mcp_handler)) as mcp_http, \
               httpx.AsyncClient(transport=httpx.MockTransport(llama_handler)) as llama_http:
        mcp_client = SportsAgenticMCPClient(http_client=mcp_http)
        llama_client = LlamaServerClient(http_client=llama_http)
        gemini_client = GeminiClient(api_key="test-gemini-key")
        router = LLMRouter(llama_client=llama_client, gemini_client=gemini_client)

        app.state.mcp_client = mcp_client
        app.state.llama_client = llama_client
        app.state.gemini_client = gemini_client
        app.state.llm_router = router
        app.state.orchestrator = AgentOrchestrator(
            mcp_client=mcp_client,
            llm_router=router,
            skills_loader=app.state.skills_loader,
            max_tool_iterations=3,
        )

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            # 1. GET /health
            health_resp = await client.get("/health")
            assert health_resp.status_code == 200
            health_data = health_resp.json()
            assert health_data["status"] == "ok"
            assert health_data["mcp_server"]["available"] is True
            assert health_data["llama_server"]["available"] is True
            assert health_data["gemini"]["available"] is True

            # 2. GET /tools
            tools_resp = await client.get("/tools")
            assert tools_resp.status_code == 200
            tools_data = tools_resp.json()
            assert len(tools_data["tools"]) == 3
            assert {t["name"] for t in tools_data["tools"]} == {
                "get_cfb_scores",
                "get_team_schedule",
                "get_cfb_rankings",
            }

            # 3. GET /skills
            skills_resp = await client.get("/skills")
            assert skills_resp.status_code == 200
            skills_data = skills_resp.json()
            assert any(s["name"] == "cfb" for s in skills_data["skills"])

            # 4. POST /chat
            chat_resp = await client.post(
                "/chat",
                json={"message": "How many wins does Ole Miss have?", "provider": "auto"},
            )
            assert chat_resp.status_code == 200
            chat_data = chat_resp.json()
            assert chat_data["reply"] == "Ole Miss has 10 wins on the season."
            assert chat_data["provider_used"] == "llama"
            assert chat_data["fallback_triggered"] is False

            # 5. POST /chat validation error on empty message
            bad_resp = await client.post("/chat", json={"message": ""})
            assert bad_resp.status_code == 422


@pytest.mark.asyncio
async def test_api_chat_returns_502_when_all_providers_fail(test_settings: Settings) -> None:
    """Verify POST /chat returns HTTP 502 when both llama-server and Gemini fail."""
    app = create_app(test_settings)

    def failing_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused")

    async with httpx.AsyncClient(transport=httpx.MockTransport(failing_handler)) as failing_http:
        mcp_client = SportsAgenticMCPClient(http_client=failing_http)
        llama_client = LlamaServerClient(http_client=failing_http)
        gemini_client = GeminiClient(api_key=None)
        router = LLMRouter(llama_client=llama_client, gemini_client=gemini_client)

        app.state.mcp_client = mcp_client
        app.state.llama_client = llama_client
        app.state.gemini_client = gemini_client
        app.state.llm_router = router
        app.state.orchestrator = AgentOrchestrator(
            mcp_client=mcp_client,
            llm_router=router,
            skills_loader=app.state.skills_loader,
        )

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            resp = await client.post("/chat", json={"message": "Who is ranked #1?"})
            assert resp.status_code == 502
            assert "Gemini backup also failed" in resp.json()["detail"]

            health_resp = await client.get("/health")
            assert health_resp.status_code == 200
            assert health_resp.json()["status"] == "degraded"
