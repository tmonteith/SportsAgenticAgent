"""Server-sent event streaming for a single agent turn."""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator, Dict, Optional, Sequence

from app.agent.orchestrator import AgentOrchestrator
from app.llm.base import LLMClientError
from app.logging_config import get_logger
from app.schemas import ConversationMessage

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
) -> AsyncIterator[str]:
    """Yield SSE frames for one agent turn, cancelling the turn if the consumer goes away.

    Inference is not token-streamed, so the reply arrives as one message.delta
    after the tool events.
    """
    queue: "asyncio.Queue[Optional[str]]" = asyncio.Queue()

    def on_tool_event(event: str, data: Dict[str, Any]) -> None:
        queue.put_nowait(format_sse(event, data))

    async def run_turn() -> None:
        try:
            response = await agent.run(
                user_message=user_message,
                history=history,
                on_tool_event=on_tool_event,
            )
            if response.reply:
                queue.put_nowait(format_sse("message.delta", {"text": response.reply}))
            queue.put_nowait(format_sse("done", {}))
        except LLMClientError as exc:
            logger.error("Streaming chat failed due to LLM error: %s", exc)
            queue.put_nowait(format_sse("error", {"message": UNAVAILABLE_MESSAGE}))
        except Exception:
            logger.exception("Streaming chat failed unexpectedly")
            queue.put_nowait(format_sse("error", {"message": UNAVAILABLE_MESSAGE}))
        finally:
            queue.put_nowait(None)

    turn = asyncio.create_task(run_turn())
    try:
        while True:
            try:
                frame = await asyncio.wait_for(queue.get(), timeout=heartbeat_seconds)
            except asyncio.TimeoutError:
                yield HEARTBEAT_COMMENT
                continue
            if frame is None:
                break
            yield frame
    finally:
        turn.cancel()
        await asyncio.gather(turn, return_exceptions=True)
