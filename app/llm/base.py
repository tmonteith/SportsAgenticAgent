"""Common data structures and response parsers for LLM inference clients."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class LLMClientError(Exception):
    """Base error raised when an LLM provider request fails."""


class LLMTimeoutError(LLMClientError):
    """Raised when an LLM provider request times out."""


class LLMConnectionError(LLMClientError):
    """Raised when an LLM provider cannot be reached or returns a server error."""


@dataclass
class ToolCallRequest:
    """Normalized representation of a tool call requested by an LLM."""

    id: str
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    extra_content: Optional[Dict[str, Any]] = None


@dataclass
class LLMResponse:
    """Normalized LLM completion response."""

    content: str
    tool_calls: List[ToolCallRequest] = field(default_factory=list)
    provider_used: str = "llama"
    fallback_triggered: bool = False


def mcp_tools_to_openai_tools(mcp_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert MCP tool definitions into OpenAI function-calling schema."""
    converted: List[Dict[str, Any]] = []
    for tool in mcp_tools:
        name = tool.get("name")
        if not name:
            continue
        parameters = tool.get("inputSchema") or tool.get("parameters") or {
            "type": "object",
            "properties": {},
        }
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool.get("description", ""),
                    "parameters": parameters,
                },
            }
        )
    return converted


def parse_openai_chat_response(
    payload: Dict[str, Any],
    provider_name: str,
    known_tool_names: Optional[List[str]] = None,
) -> LLMResponse:
    """Parse an OpenAI-compatible chat/completions payload into an LLMResponse."""
    choices = payload.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        raise LLMClientError(f"{provider_name} returned an empty or malformed choices array")

    message = choices[0].get("message") or {}
    content = message.get("content") or ""
    raw_tool_calls = message.get("tool_calls") or []

    parsed_tool_calls: List[ToolCallRequest] = []
    for idx, tc in enumerate(raw_tool_calls):
        if not isinstance(tc, dict):
            continue
        func = tc.get("function") or {}
        func_name = func.get("name")
        if not func_name:
            continue
        raw_args = func.get("arguments", "{}")
        if isinstance(raw_args, str):
            try:
                args_dict = json.loads(raw_args) if raw_args.strip() else {}
            except json.JSONDecodeError:
                args_dict = {}
        elif isinstance(raw_args, dict):
            args_dict = raw_args
        else:
            args_dict = {}

        extra_content = tc.get("extra_content")

        parsed_tool_calls.append(
            ToolCallRequest(
                id=str(tc.get("id") or f"call_{idx + 1}"),
                name=str(func_name),
                arguments=args_dict if isinstance(args_dict, dict) else {},
                extra_content=extra_content if isinstance(extra_content, dict) else None,
            )
        )

    # Fallback: if a local model emitted a JSON tool call in content instead of native tool_calls
    if not parsed_tool_calls and content and known_tool_names:
        inline_call = _try_parse_inline_tool_call(content, known_tool_names)
        if inline_call is not None:
            parsed_tool_calls.append(inline_call)
            content = ""

    return LLMResponse(
        content=content,
        tool_calls=parsed_tool_calls,
        provider_used=provider_name,
        fallback_triggered=False,
    )


def _try_parse_inline_tool_call(
    content: str,
    known_tool_names: List[str],
) -> Optional[ToolCallRequest]:
    """Extract a JSON tool call from raw assistant content if it matches a known MCP tool."""
    stripped = content.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL)
    candidate = fence_match.group(1) if fence_match else stripped

    if not (candidate.startswith("{") and candidate.endswith("}")):
        return None

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, dict):
        return None

    tool_name = parsed.get("name") or parsed.get("tool")
    if not isinstance(tool_name, str) or tool_name not in known_tool_names:
        return None

    arguments = parsed.get("arguments") or parsed.get("params") or {}
    if not isinstance(arguments, dict):
        arguments = {}

    return ToolCallRequest(
        id="call_inline_1",
        name=tool_name,
        arguments=arguments,
    )
