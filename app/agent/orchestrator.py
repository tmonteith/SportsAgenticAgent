"""Multi-step agent orchestrator binding LLM inference, MCP tools, and skills."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

from app.llm.router import LLMRouter, ProviderSelection
from app.logging_config import get_logger
from app.mcp.client import MCPClientError, SportsAgenticMCPClient
from app.schemas import ChatResponse, ConversationMessage, ToolCallRecord
from app.skills.loader import SkillsLoader

logger = get_logger("agent")


class AgentOrchestrator:
    """Coordinates system prompt construction, LLM tool calling, and MCP tool execution."""

    def __init__(
        self,
        mcp_client: SportsAgenticMCPClient,
        llm_router: LLMRouter,
        skills_loader: SkillsLoader,
        max_tool_iterations: int = 5,
    ) -> None:
        self.mcp_client = mcp_client
        self.llm_router = llm_router
        self.skills_loader = skills_loader
        self.max_tool_iterations = max_tool_iterations

    async def run(
        self,
        user_message: str,
        history: Optional[Sequence[ConversationMessage]] = None,
        provider: Optional[ProviderSelection] = None,
        skill_names: Optional[Sequence[str]] = None,
    ) -> ChatResponse:
        """Execute a full agent turn including any MCP tool calls."""
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

        for iteration in range(1, self.max_tool_iterations + 1):
            logger.debug("Agent iteration %d/%d", iteration, self.max_tool_iterations)
            llm_response = await self.llm_router.chat(
                messages=messages,
                mcp_tools=mcp_tools,
                provider=provider,
            )
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
            messages.append(
                {
                    "role": "assistant",
                    "content": llm_response.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in llm_response.tool_calls
                    ],
                }
            )

            # Execute each requested MCP tool
            for tc in llm_response.tool_calls:
                try:
                    tool_result = await self.mcp_client.call_tool(tc.name, tc.arguments)
                except MCPClientError as exc:
                    logger.warning("MCP tool '%s' failed: %s", tc.name, exc)
                    tool_result = json.dumps({"error": str(exc)})

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
                        "content": tool_result,
                    }
                )

        # Max tool iterations reached: request a final synthesis without tools
        logger.warning(
            "Reached max_tool_iterations (%d); requesting final synthesis without tools",
            self.max_tool_iterations,
        )
        final_response = await self.llm_router.chat(
            messages=messages,
            mcp_tools=None,
            provider=provider,
        )
        last_provider = final_response.provider_used
        fallback_triggered = fallback_triggered or final_response.fallback_triggered

        return ChatResponse(
            reply=final_response.content,
            provider_used=last_provider,
            fallback_triggered=fallback_triggered,
            tool_calls=executed_calls,
        )
