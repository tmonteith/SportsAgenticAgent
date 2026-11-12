"""Incremental parsing of OpenAI-compatible streamed chat completions."""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional

import httpx

from app.llm.base import (
    LLMConnectionError,
    LLMResponse,
    _try_parse_inline_tool_call,
    parse_openai_chat_response,
)

# Receives each piece of answer text that is safe to show the user.
TextDeltaCallback = Callable[[str], None]

# These mirror the fence search in _try_parse_inline_tool_call.
_FENCE_OPEN = re.compile(r"```(?:json)?\s*\{")
_FENCE_OPEN_PARTIAL = re.compile(r"```(?:j|js|jso|json)?\s*\Z")
_FENCE_BLOCK = re.compile(r"```(?:json)?\s*\{.*?\}\s*```", flags=re.DOTALL)


class ChatStreamAssembler:
    """Builds an LLMResponse from streamed chunks, passing text on as it becomes safe to show.

    Local models can write a tool call as JSON in their content, so text that could
    still turn into one is held back. The final result always comes from
    parse_openai_chat_response, so it matches what a non-streamed reply would give.
    """

    def __init__(
        self,
        provider_name: str,
        on_text_delta: TextDeltaCallback,
        known_tool_names: Optional[List[str]] = None,
    ) -> None:
        self.provider_name = provider_name
        self.on_text_delta = on_text_delta
        self.known_tool_names = known_tool_names or []
        self.finished = False
        self._content = ""
        self._sent = 0
        self._scan = 0
        self._plain_text = not self.known_tool_names
        self._tool_calls: List[Dict[str, Any]] = []
        self._calls_by_index: Dict[Optional[int], Dict[str, Any]] = {}
        self._usage: Optional[Dict[str, Any]] = None

    def feed(self, chunk: Dict[str, Any]) -> None:
        """Apply one chat.completion.chunk payload."""
        # Providers that report usage send it on a final chunk, often with no choices.
        if isinstance(chunk.get("usage"), dict):
            self._usage = chunk["usage"]

        choices = chunk.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            return

        if choices[0].get("finish_reason"):
            self.finished = True

        delta = choices[0].get("delta") or {}
        for tool_call in delta.get("tool_calls") or []:
            if isinstance(tool_call, dict):
                self._merge_tool_call(tool_call)

        text = delta.get("content")
        if text:
            self._content += text
            self._send_up_to(self._safe_end())

    def finish(self) -> LLMResponse:
        """Resolve the held text and return the completed response."""
        message: Dict[str, Any] = {"role": "assistant", "content": self._content}
        if self._tool_calls:
            message["tool_calls"] = self._tool_calls

        payload: Dict[str, Any] = {"choices": [{"message": message}]}
        if self._usage is not None:
            payload["usage"] = self._usage
        response = parse_openai_chat_response(
            payload,
            provider_name=self.provider_name,
            known_tool_names=self.known_tool_names,
        )
        if not response.tool_calls:
            self._send_up_to(len(self._content))
        return response

    def _merge_tool_call(self, delta: Dict[str, Any]) -> None:
        index = delta.get("index")
        call_id = delta.get("id")
        call = self._calls_by_index.get(index)

        # Gemini can send parallel calls under one index, so a new id starts a new call.
        if call is None or (call_id and call["id"] and call_id != call["id"]):
            call = {"id": call_id, "function": {"name": "", "arguments": ""}}
            self._tool_calls.append(call)
            self._calls_by_index[index] = call

        call["id"] = call["id"] or call_id
        function = delta.get("function") or {}
        call["function"]["name"] = call["function"]["name"] or function.get("name") or ""
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            call["function"]["arguments"] += arguments

        extra_content = delta.get("extra_content")
        if isinstance(extra_content, dict):
            call["extra_content"] = {**call.get("extra_content", {}), **extra_content}

    def _safe_end(self) -> int:
        """Return how much of the content cannot be part of an inline tool call."""
        if self._tool_calls:
            return self._sent
        if self._plain_text:
            return len(self._content)

        # A bare JSON call fills the whole reply, so it is only ruled out at the end.
        stripped = self._content.lstrip()
        if not stripped or stripped.startswith("{"):
            return self._sent

        content = self._content
        while True:
            fence = content.find("```", self._scan)
            if fence < 0:
                return len(content.rstrip("`"))

            if _FENCE_OPEN.match(content, fence):
                block = _FENCE_BLOCK.match(content, fence)
                if block is None:
                    return fence
                call = _try_parse_inline_tool_call(content[fence : block.end()], self.known_tool_names)
                if call is not None:
                    return fence
                # Only the first JSON fence is ever parsed as a call, so the rest is text.
                self._plain_text = True
                return len(content)

            if _FENCE_OPEN_PARTIAL.match(content, fence):
                return fence
            self._scan = fence + 1

    def _send_up_to(self, end: int) -> None:
        if end > self._sent:
            self.on_text_delta(self._content[self._sent : end])
            self._sent = end


async def read_chat_stream(
    response: httpx.Response,
    assembler: ChatStreamAssembler,
) -> LLMResponse:
    """Read a server-sent chat completion stream to its end."""
    provider = assembler.provider_name
    async for line in response.aiter_lines():
        # llama-server reports failures after headers as an "error:" line.
        if line.startswith("error:"):
            raise LLMConnectionError(f"{provider} reported a stream error: {line[6:].strip()}")
        if not line.startswith("data:"):
            continue

        data = line[5:].strip()
        if data == "[DONE]":
            assembler.finished = True
            break

        chunk = json.loads(data)
        if not isinstance(chunk, dict):
            continue
        if chunk.get("error"):
            raise LLMConnectionError(f"{provider} reported a stream error: {chunk['error']}")
        assembler.feed(chunk)

    if not assembler.finished:
        raise LLMConnectionError(f"{provider} stream ended before the reply was complete")
    return assembler.finish()
