"""Client for Google Gemini inference (backup and alternative provider)."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import httpx

from app.llm.base import (
    LLMClientError,
    LLMConnectionError,
    LLMResponse,
    LLMTimeoutError,
    mcp_tools_to_openai_tools,
    parse_openai_chat_response,
)
from app.logging_config import get_logger

logger = get_logger("llm.gemini")

GEMINI_BYPASS_THOUGHT_SIGNATURE = "skip_thought_signature_validator"


def sanitize_messages_for_gemini(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Ensure all assistant tool calls contain a thought_signature for Gemini 2.5/3 models."""
    sanitized: List[Dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            msg_copy = dict(msg)
            tool_calls_copy: List[Dict[str, Any]] = []
            for tc in msg["tool_calls"]:
                tc_copy = dict(tc)
                extra = tc_copy.get("extra_content")
                has_signature = False
                if isinstance(extra, dict):
                    google_meta = extra.get("google")
                    if isinstance(google_meta, dict) and google_meta.get("thought_signature"):
                        has_signature = True
                    elif extra.get("thought_signature"):
                        has_signature = True

                if not has_signature:
                    tc_copy["extra_content"] = {
                        "google": {
                            "thought_signature": GEMINI_BYPASS_THOUGHT_SIGNATURE,
                        }
                    }
                tool_calls_copy.append(tc_copy)
            msg_copy["tool_calls"] = tool_calls_copy
            sanitized.append(msg_copy)
        else:
            sanitized.append(msg)
    return sanitized



class GeminiClient:
    """Async client for Google Gemini via its OpenAI-compatible chat completions endpoint."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash",
        base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai",
        timeout_seconds: float = 30.0,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.api_key = (api_key or "").strip() or None
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._external_client = http_client

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.api_key != "your_gemini_api_key_here")

    @property
    def chat_completions_url(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        mcp_tools: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        """Send a chat completion request to Gemini."""
        if not self.is_configured:
            raise LLMClientError(
                "Gemini API key is not configured. Set the GEMINI_API_KEY environment variable."
            )

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": sanitize_messages_for_gemini(messages),
        }
        known_tools: List[str] = []
        if mcp_tools:
            openai_tools = mcp_tools_to_openai_tools(mcp_tools)
            if openai_tools:
                payload["tools"] = openai_tools
                known_tools = [
                    t["function"]["name"]
                    for t in openai_tools
                    if "function" in t and "name" in t["function"]
                ]

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        start = time.monotonic()
        logger.info("Sending inference request to Gemini (model=%s)", self.model)

        if self._external_client is not None:
            return await self._post_chat(self._external_client, payload, headers, known_tools, start)

        timeout = httpx.Timeout(self.timeout_seconds)
        async with httpx.AsyncClient(timeout=timeout) as client:
            return await self._post_chat(client, payload, headers, known_tools, start)

    async def _post_chat(
        self,
        client: httpx.AsyncClient,
        payload: Dict[str, Any],
        headers: Dict[str, str],
        known_tools: List[str],
        start: float,
    ) -> LLMResponse:
        try:
            response = await client.post(
                self.chat_completions_url,
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()
            elapsed_ms = (time.monotonic() - start) * 1000.0
            logger.info("Gemini responded in %.1fms", elapsed_ms)
            return parse_openai_chat_response(
                data,
                provider_name="gemini",
                known_tool_names=known_tools,
            )
        except httpx.TimeoutException as exc:
            logger.error("Gemini request timed out after %.1fs: %s", self.timeout_seconds, exc)
            raise LLMTimeoutError(
                f"Gemini request timed out after {self.timeout_seconds}s"
            ) from exc
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            logger.error("Gemini returned HTTP %d: %s", status, exc.response.text)
            if status >= 500:
                raise LLMConnectionError(f"Gemini returned server error HTTP {status}") from exc
            raise LLMClientError(f"Gemini returned HTTP {status}: {exc.response.text}") from exc
        except httpx.HTTPError as exc:
            logger.error("Gemini connection failed: %s", exc)
            raise LLMConnectionError(f"Could not connect to Gemini API: {exc}") from exc
        except ValueError as exc:
            logger.error("Invalid JSON from Gemini: %s", exc)
            raise LLMClientError(f"Invalid JSON from Gemini: {exc}") from exc

    async def check_health(self) -> tuple[bool, str]:
        """Report whether Gemini is configured and ready as backup/alternative."""
        if self.is_configured:
            return True, f"Configured (model={self.model})"
        return False, "GEMINI_API_KEY not set"
