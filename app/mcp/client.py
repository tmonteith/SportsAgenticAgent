"""Async MCP client for the SportsAgentic SSE + JSON-RPC 2.0 server."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse, urlunparse

import httpx

from app.logging_config import get_logger

logger = get_logger("mcp")


class MCPClientError(Exception):
    """Raised when communicating with the SportsAgentic MCP server fails."""


DEFAULT_CFB_TOOLS: List[Dict[str, Any]] = [
    {
        "name": "get_cfb_scores",
        "description": (
            "Fetch College Football scores, game schedules, and match results. "
            "IMPORTANT: Always supply the 'team' argument when a specific team or game is requested "
            "to avoid retrieving hundreds of nationwide games."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "year": {
                    "type": "number",
                    "description": "Season year (defaults to current year)",
                },
                "week": {
                    "type": "number",
                    "description": "Week number (1-15)",
                },
                "date": {
                    "type": "string",
                    "description": "Specific date filter formatted as YYYY-MM-DD",
                },
                "team": {
                    "type": "string",
                    "description": (
                        "Team name or abbreviation (e.g., 'CSU', 'Ole Miss', 'Bama', 'Ohio State'). "
                        "ALWAYS provide this if the question is about a specific team."
                    ),
                },
            },
        },
    },
    {
        "name": "get_team_schedule",
        "description": (
            "Get past or upcoming schedule and game results for a specific "
            "College Football team."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "team": {
                    "type": "string",
                    "description": (
                        "Team name or abbreviation "
                        "(e.g., 'Georgia', 'Michigan', 'Texas', 'CSU')"
                    ),
                },
                "year": {
                    "type": "number",
                    "description": "Season year (defaults to current year)",
                },
            },
            "required": ["team"],
        },
    },
    {
        "name": "get_cfb_rankings",
        "description": (
            "Get national AP Top 25 or CFP rankings for a given season week."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "year": {
                    "type": "number",
                    "description": "Season year (defaults to current year)",
                },
                "week": {
                    "type": "number",
                    "description": "Week number",
                },
            },
        },
    },
]


def _resolve_session_endpoint(sse_url: str, endpoint_url: str) -> str:
    """Resolve session endpoint URL while preserving the scheme, host, and port of sse_url."""
    parsed_sse = urlparse(sse_url)
    joined = urljoin(sse_url, endpoint_url)
    parsed_joined = urlparse(joined)

    return urlunparse(
        (
            parsed_sse.scheme,
            parsed_sse.netloc,
            parsed_joined.path,
            parsed_joined.params,
            parsed_joined.query,
            parsed_joined.fragment,
        )
    )


class SportsAgenticMCPClient:
    """Client for interacting with the SportsAgentic Go MCP server over SSE."""

    def __init__(
        self,
        sse_url: str = "http://localhost:8080/sse",
        timeout_seconds: float = 10.0,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.sse_url = sse_url
        self.timeout_seconds = timeout_seconds
        self._external_client = http_client
        self._request_id = 0

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    async def _send_rpc(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Connect to the SSE endpoint, discover the session message URL, and execute a JSON-RPC call."""
        rpc_id = self._next_id()
        payload: Dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "method": method,
            "params": params or {},
        }

        start_time = time.monotonic()
        logger.debug("Opening MCP SSE connection to %s for method=%s", self.sse_url, method)

        if self._external_client is not None:
            return await self._execute_session_rpc(self._external_client, payload, start_time)

        timeout = httpx.Timeout(self.timeout_seconds)
        async with httpx.AsyncClient(timeout=timeout) as client:
            return await self._execute_session_rpc(client, payload, start_time)

    async def _execute_session_rpc(
        self,
        client: httpx.AsyncClient,
        payload: Dict[str, Any],
        start_time: float,
    ) -> Dict[str, Any]:
        """Perform the SSE endpoint handshake and POST the JSON-RPC request while the stream is active."""
        method = payload.get("method", "unknown")
        rpc_id = payload.get("id")

        try:
            async with client.stream("GET", self.sse_url) as sse_response:
                sse_response.raise_for_status()
                lines_iter = sse_response.aiter_lines()

                endpoint_url = await self._read_endpoint_from_sse(lines_iter)
                if not endpoint_url:
                    raise MCPClientError(
                        f"Did not receive a session message endpoint from MCP SSE stream at {self.sse_url}"
                    )

                resolved_endpoint = _resolve_session_endpoint(self.sse_url, endpoint_url)
                logger.debug("MCP session endpoint resolved to %s", resolved_endpoint)

                post_resp = await client.post(
                    resolved_endpoint,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                post_resp.raise_for_status()

                body_text = post_resp.text.strip()
                if body_text:
                    data = json.loads(body_text)
                else:
                    # If server returned 202 Accepted with empty body, read JSON-RPC reply from SSE stream
                    data = await self._read_rpc_response_from_sse(lines_iter, rpc_id)

                elapsed_ms = (time.monotonic() - start_time) * 1000.0
                logger.info("MCP %s completed in %.1fms", method, elapsed_ms)

                if "error" in data and data["error"]:
                    err_obj = data["error"]
                    err_msg = (
                        err_obj.get("message", str(err_obj))
                        if isinstance(err_obj, dict)
                        else str(err_obj)
                    )
                    raise MCPClientError(f"MCP error in {method}: {err_msg}")

                return data

        except MCPClientError:
            raise
        except httpx.TimeoutException as exc:
            logger.error("Timed out communicating with SportsAgentic MCP at %s: %s", self.sse_url, exc)
            raise MCPClientError(
                f"Timed out connecting to SportsAgentic MCP server at {self.sse_url}"
            ) from exc
        except httpx.HTTPError as exc:
            logger.error("HTTP error communicating with SportsAgentic MCP at %s: %s", self.sse_url, exc)
            raise MCPClientError(
                f"Failed to communicate with SportsAgentic MCP server at {self.sse_url}: {exc}"
            ) from exc
        except json.JSONDecodeError as exc:
            logger.error("Invalid JSON returned by SportsAgentic MCP: %s", exc)
            raise MCPClientError(f"Invalid JSON response from MCP server: {exc}") from exc

    async def _read_endpoint_from_sse(self, lines_iter: Any) -> Optional[str]:
        """Read lines from the SSE stream until the first data: endpoint line arrives."""
        current_event: Optional[str] = None
        while True:
            try:
                raw_line = await asyncio.wait_for(
                    lines_iter.__anext__(),
                    timeout=self.timeout_seconds,
                )
            except StopAsyncIteration:
                return None
            except asyncio.TimeoutError as exc:
                raise MCPClientError(
                    f"Timed out waiting for SSE endpoint event from {self.sse_url}"
                ) from exc

            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("event:"):
                current_event = line.split("event:", 1)[1].strip()
            elif line.startswith("data:"):
                data_val = line.split("data:", 1)[1].strip()
                if current_event in (None, "endpoint") and data_val:
                    return data_val

    async def _read_rpc_response_from_sse(
        self,
        lines_iter: Any,
        expected_id: Any,
    ) -> Dict[str, Any]:
        """Read SSE data events until the matching JSON-RPC response arrives."""
        while True:
            try:
                raw_line = await asyncio.wait_for(
                    lines_iter.__anext__(),
                    timeout=self.timeout_seconds,
                )
            except StopAsyncIteration as exc:
                raise MCPClientError(
                    "SSE stream closed before JSON-RPC response was received"
                ) from exc
            except asyncio.TimeoutError as exc:
                raise MCPClientError(
                    "Timed out waiting for JSON-RPC response on SSE stream"
                ) from exc

            line = raw_line.strip()
            if line.startswith("data:"):
                raw_data = line.split("data:", 1)[1].strip()
                if not raw_data:
                    continue
                try:
                    parsed = json.loads(raw_data)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict) and (
                    parsed.get("id") == expected_id or "result" in parsed or "error" in parsed
                ):
                    return parsed

    async def list_tools(self, fallback_to_defaults: bool = True) -> List[Dict[str, Any]]:
        """Query available tools from SportsAgentic via tools/list."""
        try:
            response = await self._send_rpc("tools/list", {})
            tools = response.get("result", {}).get("tools", [])
            if isinstance(tools, list) and tools:
                logger.info("Discovered %d MCP tools from SportsAgentic", len(tools))
                return tools
            if fallback_to_defaults:
                logger.warning("Empty tools list from MCP server; using built-in CFB tool schemas")
                return list(DEFAULT_CFB_TOOLS)
            return []
        except MCPClientError as exc:
            if fallback_to_defaults:
                logger.warning(
                    "Could not fetch live tools/list from MCP (%s); using built-in CFB tool schemas",
                    exc,
                )
                return list(DEFAULT_CFB_TOOLS)
            raise

    async def call_tool(self, tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> str:
        """Execute an MCP tool via tools/call and return its text payload."""
        safe_args = arguments or {}
        logger.info("Calling MCP tool '%s' with arguments=%s", tool_name, json.dumps(safe_args))
        response = await self._send_rpc(
            "tools/call",
            {
                "name": tool_name,
                "arguments": safe_args,
            },
        )
        result = response.get("result", {})
        if isinstance(result, dict):
            is_error = result.get("isError", False)
            content_list = result.get("content", [])
            texts: List[str] = []
            if isinstance(content_list, list):
                for item in content_list:
                    if isinstance(item, dict) and item.get("type") == "text":
                        texts.append(str(item.get("text", "")))
            combined = "\n".join(t for t in texts if t)
            if is_error:
                logger.warning("MCP tool '%s' returned tool-level error: %s", tool_name, combined)
                raise MCPClientError(combined or f"MCP tool '{tool_name}' returned an error")
            if combined:
                return combined
        return json.dumps(result)

    async def check_health(self) -> tuple[bool, str]:
        """Check whether the SportsAgentic MCP SSE endpoint is reachable."""
        try:
            tools = await self.list_tools(fallback_to_defaults=False)
            return True, f"Connected ({len(tools)} tools available)"
        except MCPClientError as exc:
            return False, str(exc)
