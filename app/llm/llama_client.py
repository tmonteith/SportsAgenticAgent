"""Client for local llama-server (llama.cpp OpenAI-compatible HTTP server)."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, urlunparse

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

logger = get_logger("llm.llama")


class LlamaServerClient:
    """Async client for llama-server inference."""

    def __init__(
        self,
        base_url: str = "http://localhost:8081/v1",
        model: str = "local-model",
        timeout_seconds: float = 15.0,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self._external_client = http_client

    @property
    def chat_completions_url(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        if self.base_url.endswith("/v1"):
            return f"{self.base_url}/chat/completions"
        return f"{self.base_url}/v1/chat/completions"

    @property
    def health_url(self) -> str:
        parsed = urlparse(self.base_url)
        return urlunparse((parsed.scheme, parsed.netloc, "/health", "", "", ""))

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        mcp_tools: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        """Send a chat completion request to llama-server."""
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
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

        start = time.monotonic()
        logger.info("Sending inference request to llama-server (%s)", self.chat_completions_url)

        if self._external_client is not None:
            return await self._post_chat(self._external_client, payload, known_tools, start)

        timeout = httpx.Timeout(self.timeout_seconds)
        async with httpx.AsyncClient(timeout=timeout) as client:
            return await self._post_chat(client, payload, known_tools, start)

    async def _post_chat(
        self,
        client: httpx.AsyncClient,
        payload: Dict[str, Any],
        known_tools: List[str],
        start: float,
    ) -> LLMResponse:
        try:
            response = await client.post(
                self.chat_completions_url,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            data = response.json()
            elapsed_ms = (time.monotonic() - start) * 1000.0
            logger.info("llama-server responded in %.1fms", elapsed_ms)
            return parse_openai_chat_response(
                data,
                provider_name="llama",
                known_tool_names=known_tools,
            )
        except httpx.TimeoutException as exc:
            logger.warning("llama-server timed out after %.1fs: %s", self.timeout_seconds, exc)
            raise LLMTimeoutError(
                f"llama-server timed out after {self.timeout_seconds}s"
            ) from exc
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            logger.warning("llama-server returned HTTP %d: %s", status, exc.response.text)
            if status >= 500:
                raise LLMConnectionError(
                    f"llama-server returned server error HTTP {status}"
                ) from exc
            raise LLMClientError(f"llama-server returned HTTP {status}: {exc.response.text}") from exc
        except httpx.HTTPError as exc:
            logger.warning("llama-server connection failed: %s", exc)
            raise LLMConnectionError(f"Could not connect to llama-server: {exc}") from exc
        except ValueError as exc:
            logger.error("Invalid JSON from llama-server: %s", exc)
            raise LLMClientError(f"Invalid JSON from llama-server: {exc}") from exc

    async def check_health(self) -> tuple[bool, str]:
        """Probe llama-server /health endpoint."""
        try:
            if self._external_client is not None:
                resp = await self._external_client.get(self.health_url)
            else:
                async with httpx.AsyncClient(timeout=httpx.Timeout(3.0)) as client:
                    resp = await client.get(self.health_url)
            resp.raise_for_status()
            return True, f"Reachable at {self.health_url}"
        except httpx.HTTPError as exc:
            return False, f"Unreachable ({exc})"
