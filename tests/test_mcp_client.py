"""Unit tests for the SportsAgentic MCP SSE + JSON-RPC client."""

from __future__ import annotations

import json
import httpx
import pytest

from app.mcp.client import DEFAULT_CFB_TOOLS, MCPClientError, SportsAgenticMCPClient, _resolve_session_endpoint


def test_resolve_session_endpoint_cross_host_override() -> None:
    """Verify session endpoint overrides localhost in SSE data payload with the target host."""
    # Go server returning localhost:8080 when agent connects via host.docker.internal
    resolved = _resolve_session_endpoint(
        "http://host.docker.internal:8080/sse",
        "http://localhost:8080/message?sessionId=abc-123",
    )
    assert resolved == "http://host.docker.internal:8080/message?sessionId=abc-123"

    # Go server returning localhost:8080 when agent connects via Docker network alias
    resolved_docker = _resolve_session_endpoint(
        "http://sportsagentic:8080/sse",
        "http://localhost:8080/message?sessionId=xyz-789",
    )
    assert resolved_docker == "http://sportsagentic:8080/message?sessionId=xyz-789"



@pytest.mark.asyncio
async def test_mcp_list_tools_and_call_tool_inline_response() -> None:
    """Verify SSE endpoint discovery, tools/list, and tools/call with inline JSON-RPC responses."""
    captured_posts = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/sse":
            sse_body = "event: endpoint\ndata: http://localhost:8080/message?sessionId=sess-123\n\n"
            return httpx.Response(
                200,
                content=sse_body.encode("utf-8"),
                headers={"Content-Type": "text/event-stream"},
            )
        if request.url.path == "/message":
            body = json.loads(request.content.decode("utf-8"))
            captured_posts.append(body)
            method = body["method"]
            rpc_id = body["id"]
            if method == "tools/list":
                return httpx.Response(
                    200,
                    json={
                        "jsonrpc": "2.0",
                        "id": rpc_id,
                        "result": {"tools": DEFAULT_CFB_TOOLS},
                    },
                )
            if method == "tools/call":
                return httpx.Response(
                    200,
                    json={
                        "jsonrpc": "2.0",
                        "id": rpc_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps(
                                        [{"home_team": "Ole Miss", "home_points": 38, "away_team": "LSU", "away_points": 24}]
                                    ),
                                }
                            ]
                        },
                    },
                )
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        mcp = SportsAgenticMCPClient(
            sse_url="http://localhost:8080/sse",
            http_client=http_client,
        )
        tools = await mcp.list_tools(fallback_to_defaults=False)
        assert len(tools) == 3
        assert tools[0]["name"] == "get_cfb_scores"

        result_text = await mcp.call_tool("get_cfb_scores", {"team": "Ole Miss"})
        parsed = json.loads(result_text)
        assert parsed[0]["home_team"] == "Ole Miss"
        assert parsed[0]["home_points"] == 38

        ok, detail = await mcp.check_health()
        assert ok is True
        assert "3 tools available" in detail

    assert len(captured_posts) == 3
    assert captured_posts[1]["params"] == {"name": "get_cfb_scores", "arguments": {"team": "Ole Miss"}}


@pytest.mark.asyncio
async def test_mcp_call_tool_via_sse_stream_event() -> None:
    """Verify MCP client supports servers that reply with 202 on POST and emit JSON-RPC on SSE stream."""
    rpc_reply = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "content": [{"type": "text", "text": "Rank 1: Georgia"}]
            },
        }
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/sse":
            sse_body = (
                "event: endpoint\n"
                "data: /message?sessionId=sess-456\n\n"
                "event: message\n"
                f"data: {rpc_reply}\n\n"
            )
            return httpx.Response(200, content=sse_body.encode("utf-8"))
        if request.url.path == "/message":
            return httpx.Response(202, content=b"")
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        mcp = SportsAgenticMCPClient(
            sse_url="http://localhost:8080/sse",
            http_client=http_client,
        )
        output = await mcp.call_tool("get_cfb_rankings", {"week": 7})
        assert output == "Rank 1: Georgia"


@pytest.mark.asyncio
async def test_mcp_error_handling_and_fallback_tools() -> None:
    """Verify connection failures fall back to DEFAULT_CFB_TOOLS when requested and raise on tool calls."""

    def failing_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused")

    transport = httpx.MockTransport(failing_handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        mcp = SportsAgenticMCPClient(
            sse_url="http://localhost:8080/sse",
            http_client=http_client,
        )

        # Fallback enabled
        fallback_tools = await mcp.list_tools(fallback_to_defaults=True)
        assert len(fallback_tools) == 3

        # Fallback disabled
        with pytest.raises(MCPClientError, match="Failed to communicate"):
            await mcp.list_tools(fallback_to_defaults=False)

        ok, detail = await mcp.check_health()
        assert ok is False
        assert "Failed to communicate" in detail


@pytest.mark.asyncio
async def test_mcp_tool_is_error_raises() -> None:
    """Verify MCP tool-level errors (isError=True) raise MCPClientError."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/sse":
            return httpx.Response(200, content=b"data: /message?sessionId=err\n\n")
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "isError": True,
                    "content": [{"type": "text", "text": "team parameter is required"}],
                },
            },
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        mcp = SportsAgenticMCPClient(http_client=http_client)
        with pytest.raises(MCPClientError, match="team parameter is required"):
            await mcp.call_tool("get_team_schedule", {})
