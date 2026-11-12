"""Multi-step agent orchestrator binding LLM inference, MCP tools, and skills."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable, Dict, List, Optional, Sequence

import httpx

from app.agent.tool_results import RANKINGS_TOOL, compact_tool_result, needs_rankings
from app.llm.base import LLMResponse
from app.llm.router import LLMRouter, ProviderSelection
from app.llm.streaming import TextDeltaCallback
from app.logging_config import get_logger
from app.mcp.client import MCPClientError, SportsAgenticMCPClient
from app.schemas import ChatResponse, ConversationMessage, ToolCallRecord
from app.skills.loader import SkillsLoader
from app.telemetry.events import (
    ERROR_CLIENT_DISCONNECTED,
    ERROR_INTERNAL,
    ERROR_MCP,
    ERROR_MCP_TIMEOUT,
)
from app.telemetry.request import current_request

logger = get_logger("agent")

# Receives ("tool.start" | "tool.end", payload) as tools run during a turn.
ToolEventCallback = Callable[[str, Dict[str, Any]], None]

# Later results in a turn still get this much, however much earlier ones used.
MIN_TOOL_RESULT_CHARS = 800


class AgentOrchestrator:
    """Coordinates system prompt construction, LLM tool calling, and MCP tool execution."""

    def __init__(
        self,
        mcp_client: SportsAgenticMCPClient,
        llm_router: LLMRouter,
        skills_loader: SkillsLoader,
        max_tool_iterations: int = 5,
        tool_result_max_chars: int = 2400,
    ) -> None:
        self.mcp_client = mcp_client
        self.llm_router = llm_router
        self.skills_loader = skills_loader
        self.max_tool_iterations = max_tool_iterations
        self.tool_result_max_chars = tool_result_max_chars

    async def _call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Call an MCP tool, reporting a tool_called event when telemetry is on."""
        telemetry = current_request.get()
        if telemetry is None:
            return await self.mcp_client.call_tool(tool_name, arguments)

        timer = telemetry.tool_started()
        try:
            result = await self.mcp_client.call_tool(tool_name, arguments)
        except MCPClientError as exc:
            telemetry.tool_called(tool_name, timer, _mcp_error_code(exc))
            raise
        except asyncio.CancelledError:
            telemetry.tool_called(tool_name, timer, ERROR_CLIENT_DISCONNECTED)
            raise
        except Exception:
            telemetry.tool_called(tool_name, timer, ERROR_INTERNAL)
            raise
        telemetry.tool_called(tool_name, timer)
        return result

    async def _compact_tool_result(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result: str,
        max_chars: int,
    ) -> str:
        """Shrink a tool result for the model, adding poll ranks to nationwide scores."""
        rankings: Optional[str] = None
        if needs_rankings(tool_name, arguments, result):
            year_args = {"year": arguments["year"]} if "year" in arguments else {}
            try:
                rankings = await self._call_tool(RANKINGS_TOOL, year_args)
            except MCPClientError as exc:
                logger.warning("Could not fetch rankings for nationwide scores: %s", exc)

        compacted = compact_tool_result(
            tool_name,
            arguments,
            result,
            max_chars=max_chars,
            rankings=rankings,
        )
        logger.info(
            "Compacted '%s' result from %d to %d chars",
            tool_name,
            len(result),
            len(compacted),
        )
        return compacted

    async def run(
        self,
        user_message: str,
        history: Optional[Sequence[ConversationMessage]] = None,
        provider: Optional[ProviderSelection] = None,
        skill_names: Optional[Sequence[str]] = None,
        on_tool_event: Optional[ToolEventCallback] = None,
        on_text_delta: Optional[TextDeltaCallback] = None,
    ) -> ChatResponse:
        """Execute a full agent turn including any MCP tool calls.

        With on_text_delta, inference is streamed and text is reported as it arrives.
        """
        system_prompt = self.skills_loader.build_system_prompt(skill_names=skill_names)
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
        ]

        if history:
            for item in history:
                msg_dict: Dict[str, Any] = {
                    "role": item.role,
                    "content": item.content,
                }
                if item.name:
                    msg_dict["name"] = item.name
                if item.tool_call_id:
                    msg_dict["tool_call_id"] = item.tool_call_id
                messages.append(msg_dict)

        messages.append({"role": "user", "content": user_message})

        mcp_tools = await self.mcp_client.list_tools(fallback_to_defaults=True)
        executed_calls: List[ToolCallRecord] = []
        last_provider = "llama"
        fallback_triggered = False
        text_sent = False
        needs_break = False
        tool_chars_left = self.tool_result_max_chars

        def send_text(text: str) -> None:
            nonlocal text_sent, needs_break
            # Clients append deltas, so keep text from separate iterations apart.
            on_text_delta(f"\n\n{text}" if needs_break else text)
            text_sent = True
            needs_break = False

        async def infer(tools: Optional[List[Dict[str, Any]]]) -> LLMResponse:
            if on_text_delta is None:
                response = await self.llm_router.chat(
                    messages=messages,
                    mcp_tools=tools,
                    provider=provider,
                )
            else:
                response = await self.llm_router.chat_stream(
                    messages=messages,
                    on_text_delta=send_text,
                    mcp_tools=tools,
                    provider=provider,
                    allow_fallback=not text_sent,
                )
            telemetry = current_request.get()
            if telemetry is not None:
                telemetry.record_llm_response(response)
            return response

        for iteration in range(1, self.max_tool_iterations + 1):
            logger.debug("Agent iteration %d/%d", iteration, self.max_tool_iterations)
            llm_response = await infer(mcp_tools)
            last_provider = llm_response.provider_used
            fallback_triggered = fallback_triggered or llm_response.fallback_triggered

            if not llm_response.tool_calls:
                logger.info(
                    "Agent completed turn in %d iteration(s) via provider='%s' (fallback=%s, tools_called=%d)",
                    iteration,
                    last_provider,
                    fallback_triggered,
                    len(executed_calls),
                )
                return ChatResponse(
                    reply=llm_response.content,
                    provider_used=last_provider,
                    fallback_triggered=fallback_triggered,
                    tool_calls=executed_calls,
                )

            # Record assistant tool_calls message in conversation history
            assistant_tool_calls: List[Dict[str, Any]] = []
            for tc in llm_response.tool_calls:
                call_dict: Dict[str, Any] = {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                if tc.extra_content:
                    call_dict["extra_content"] = tc.extra_content
                assistant_tool_calls.append(call_dict)

            messages.append(
                {
                    "role": "assistant",
                    "content": llm_response.content or "",
                    "tool_calls": assistant_tool_calls,
                }
            )

            # Execute each requested MCP tool
            for tc in llm_response.tool_calls:
                # Provider call ids repeat across iterations, so qualify them per turn.
                event_id = f"{iteration}:{tc.id}"
                if on_tool_event:
                    on_tool_event(
                        "tool.start",
                        {"id": event_id, "name": tc.name, "args": tc.arguments},
                    )

                tool_ok = True
                try:
                    tool_result = await self._call_tool(tc.name, tc.arguments)
                except MCPClientError as exc:
                    logger.warning("MCP tool '%s' failed: %s", tc.name, exc)
                    tool_result = json.dumps({"error": str(exc)})
                    tool_ok = False

                model_result = tool_result
                if tool_ok:
                    model_result = await self._compact_tool_result(
                        tc.name,
                        tc.arguments,
                        tool_result,
                        max(tool_chars_left, MIN_TOOL_RESULT_CHARS),
                    )
                tool_chars_left -= len(model_result)

                if on_tool_event:
                    on_tool_event(
                        "tool.end",
                        {"id": event_id, "name": tc.name, "ok": tool_ok},
                    )

                executed_calls.append(
                    ToolCallRecord(
                        name=tc.name,
                        arguments=tc.arguments,
                        result=tool_result,
                    )
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tc.name,
                        "content": model_result,
                    }
                )

            needs_break = text_sent

        # Max tool iterations reached: request a final synthesis without tools
        logger.warning(
            "Reached max_tool_iterations (%d); requesting final synthesis without tools",
            self.max_tool_iterations,
        )
        final_response = await infer(None)
        last_provider = final_response.provider_used
        fallback_triggered = fallback_triggered or final_response.fallback_triggered

        return ChatResponse(
            reply=final_response.content,
            provider_used=last_provider,
            fallback_triggered=fallback_triggered,
            tool_calls=executed_calls,
        )


def _mcp_error_code(exc: MCPClientError) -> str:
    """mcp_timeout when the MCP call timed out, else mcp_error."""
    cause = exc.__cause__
    if isinstance(cause, (httpx.TimeoutException, asyncio.TimeoutError)):
        return ERROR_MCP_TIMEOUT
    return ERROR_MCP
