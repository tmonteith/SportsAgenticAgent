"""Server-sent event streaming for a single agent turn."""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator, Dict, Optional, Sequence

from app.agent.orchestrator import AgentOrchestrator
from app.llm.base import LLMClientError
from app.logging_config import get_logger
from app.schemas import ConversationMessage
from app.telemetry.events import ERROR_CLIENT_DISCONNECTED, ERROR_INTERNAL, ERROR_LLM
from app.telemetry.request import RequestTelemetry, current_request

logger = get_logger("agent.streaming")

HEARTBEAT_SECONDS = 15.0
HEARTBEAT_COMMENT = ": ping\n\n"
UNAVAILABLE_MESSAGE = "The assistant is temporarily unavailable. Please try again."


def format_sse(event: str, data: Dict[str, Any]) -> str:
    """Serialize one server-sent event."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def stream_agent_turn(
    agent: AgentOrchestrator,
    user_message: str,
    history: Optional[Sequence[ConversationMessage]] = None,
    heartbeat_seconds: float = HEARTBEAT_SECONDS,
    telemetry: Optional[RequestTelemetry] = None,
) -> AsyncIterator[str]:
    """Yield SSE frames for one agent turn, cancelling the turn if the consumer goes away.

    With telemetry, the turn counts as completed once the last frame has been handed
    to the server. A stream abandoned before that is request_failed with
    client_disconnected, unless the turn had already failed for another reason.
    """
    queue: "asyncio.Queue[Optional[str]]" = asyncio.Queue()
    text_sent = False
    error_code: Optional[str] = None
    stream_ended = False

    def on_tool_event(event: str, data: Dict[str, Any]) -> None:
        queue.put_nowait(format_sse(event, data))

    def on_text_delta(text: str) -> None:
        nonlocal text_sent
        text_sent = True
        queue.put_nowait(format_sse("message.delta", {"text": text}))

    async def run_turn() -> None:
        nonlocal error_code
        if telemetry is not None:
            # This task runs in its own copy of the context, so the orchestrator sees it.
            current_request.set(telemetry)
        try:
            response = await agent.run(
                user_message=user_message,
                history=history,
                on_tool_event=on_tool_event,
                on_text_delta=on_text_delta,
            )
            if response.reply and not text_sent:
                on_text_delta(response.reply)
            queue.put_nowait(format_sse("done", {}))
        except LLMClientError as exc:
            error_code = ERROR_LLM
            logger.error("Streaming chat failed due to LLM error: %s", exc)
            queue.put_nowait(format_sse("error", {"message": UNAVAILABLE_MESSAGE}))
        except Exception:
            error_code = ERROR_INTERNAL
            logger.exception("Streaming chat failed unexpectedly")
            queue.put_nowait(format_sse("error", {"message": UNAVAILABLE_MESSAGE}))
        finally:
            queue.put_nowait(None)

    if telemetry is not None:
        telemetry.start()
    turn = asyncio.create_task(run_turn())
    try:
        while True:
            try:
                frame = await asyncio.wait_for(queue.get(), timeout=heartbeat_seconds)
            except asyncio.TimeoutError:
                yield HEARTBEAT_COMMENT
                continue
            if frame is None:
                stream_ended = True
                break
            yield frame
    finally:
        turn.cancel()
        try:
            await asyncio.gather(turn, return_exceptions=True)
        finally:
            if telemetry is not None:
                if stream_ended and error_code is None:
                    telemetry.complete()
                else:
                    telemetry.fail(error_code or ERROR_CLIENT_DISCONNECTED)
