"""LLM router providing automatic failover from llama-server to Gemini."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from app.llm.base import LLMClientError, LLMResponse
from app.llm.gemini_client import GeminiClient
from app.llm.llama_client import LlamaServerClient
from app.logging_config import get_logger

logger = get_logger("llm.router")

ProviderSelection = Literal["auto", "llama", "gemini"]


class LLMRouter:
    """Routes inference requests to llama-server with automatic backup to Gemini."""

    def __init__(
        self,
        llama_client: LlamaServerClient,
        gemini_client: GeminiClient,
        default_provider: ProviderSelection = "auto",
    ) -> None:
        self.llama_client = llama_client
        self.gemini_client = gemini_client
        self.default_provider: ProviderSelection = default_provider

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        mcp_tools: Optional[List[Dict[str, Any]]] = None,
        provider: Optional[ProviderSelection] = None,
    ) -> LLMResponse:
        """Execute a chat completion using the selected provider or auto failover."""
        selected: ProviderSelection = provider or self.default_provider

        if selected == "gemini":
            logger.info("Routing inference directly to Gemini (explicit provider='gemini')")
            response = await self.gemini_client.chat(messages, mcp_tools=mcp_tools)
            response.provider_used = "gemini"
            response.fallback_triggered = False
            return response

        if selected == "llama":
            logger.info("Routing inference directly to llama-server (explicit provider='llama')")
            response = await self.llama_client.chat(messages, mcp_tools=mcp_tools)
            response.provider_used = "llama"
            response.fallback_triggered = False
            return response

        # Default "auto" mode: try llama-server first, fall back to Gemini if down or timing out
        try:
            response = await self.llama_client.chat(messages, mcp_tools=mcp_tools)
            response.provider_used = "llama"
            response.fallback_triggered = False
            return response
        except LLMClientError as llama_err:
            logger.warning(
                "Primary llama-server failed (%s); failing over to Gemini backup",
                llama_err,
            )
            try:
                fallback_resp = await self.gemini_client.chat(messages, mcp_tools=mcp_tools)
                fallback_resp.provider_used = "gemini"
                fallback_resp.fallback_triggered = True
                logger.info("Gemini backup succeeded after llama-server failure")
                return fallback_resp
            except LLMClientError as gemini_err:
                logger.error(
                    "Both llama-server and Gemini backup failed. llama=%s | gemini=%s",
                    llama_err,
                    gemini_err,
                )
                raise LLMClientError(
                    f"Primary llama-server failed ({llama_err}) and Gemini backup also failed ({gemini_err})"
                ) from gemini_err
