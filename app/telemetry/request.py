"""Per-request telemetry: builds the agent.events for one agent turn."""

from __future__ import annotations

from contextvars import ContextVar
from datetime import datetime
import time
from typing import Any, Callable, Dict, Mapping, Optional, Protocol

from app.logging_config import get_logger
from app.telemetry.events import (
    REQUEST_COMPLETED,
    REQUEST_FAILED,
    REQUEST_STARTED,
    REQUEST_STATUS_COMPLETED,
    REQUEST_STATUS_FAILED,
    TOOL_CALLED,
    TOOL_STATUS_ERROR,
    TOOL_STATUS_OK,
    new_id,
    rfc3339_utc,
    utc_now,
)

logger = get_logger("telemetry")


class EventSink(Protocol):
    def emit(self, event: Mapping[str, Any]) -> None: ...


class ToolTimer:
    """When a tool call started, by wall clock (for event_time) and monotonic clock (for latency)."""

    __slots__ = ("wall", "mono")

    def __init__(self, wall: datetime, mono: float) -> None:
        self.wall = wall
        self.mono = mono


class RequestTelemetry:
    """Emits request_started, tool_called and one terminal event for an agent turn.

    Every method swallows its own errors: telemetry must never break a request.
    """

    def __init__(
        self,
        sink: EventSink,
        request_id: str,
        user_key: str,
        session_id: Optional[str],
        client: Optional[str],
        app_version: Optional[str],
        prompt_chars: int,
        model_names: Optional[Mapping[str, str]] = None,
        clock: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] = utc_now,
    ) -> None:
        self._sink = sink
        self.request_id = request_id
        self.user_key = user_key
        self.session_id = session_id
        self.client = client
        self.app_version = app_version
        self.prompt_chars = prompt_chars
        self._model_names = dict(model_names or {})
        self._clock = clock
        self._now = now

        self._started_mono: Optional[float] = None
        self.model: Optional[str] = None
        self.input_tokens: Optional[int] = None
        self.output_tokens: Optional[int] = None
        self.started = False
        self.finished = False

    def start(self) -> None:
        """Emit request_started (once) and start the latency clock."""
        if self.started:
            return
        self.started = True
        self._started_mono = self._clock()
        self._emit(
            REQUEST_STARTED,
            self._now(),
            prompt_chars=self.prompt_chars,
            client=self.client,
            app_version=self.app_version,
        )

    def tool_started(self) -> ToolTimer:
        return ToolTimer(self._now(), self._clock())

    def tool_called(self, tool: str, timer: ToolTimer, error_code: Optional[str] = None) -> None:
        """Emit tool_called, timed from timer. event_time is when the call started."""
        self._emit(
            TOOL_CALLED,
            timer.wall,
            tool=tool,
            latency_ms=self._elapsed_ms(timer.mono),
            status=TOOL_STATUS_ERROR if error_code else TOOL_STATUS_OK,
            error_code=error_code,
        )

    def record_llm_response(self, response: Any) -> None:
        """Note the model used and add the tokens it reported, if any."""
        try:
            provider = getattr(response, "provider_used", None)
            if isinstance(provider, str) and provider in self._model_names:
                self.model = self._model_names[provider]
            self.input_tokens = _add_tokens(self.input_tokens, getattr(response, "input_tokens", None))
            self.output_tokens = _add_tokens(self.output_tokens, getattr(response, "output_tokens", None))
        except Exception:  # pragma: no cover - defensive
            logger.debug("Could not record LLM usage for telemetry", exc_info=True)

    def complete(self) -> None:
        """Emit request_completed, unless the request already ended."""
        self._finish(REQUEST_COMPLETED, REQUEST_STATUS_COMPLETED, None)

    def fail(self, error_code: str) -> None:
        """Emit request_failed with error_code, unless the request already ended."""
        self._finish(REQUEST_FAILED, REQUEST_STATUS_FAILED, error_code)

    def _finish(self, event_type: str, status: str, error_code: Optional[str]) -> None:
        if self.finished:
            return
        self.start()
        self.finished = True
        self._emit(
            event_type,
            self._now(),
            latency_ms=self._elapsed_ms(self._started_mono),
            status=status,
            error_code=error_code,
            model=self.model,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            prompt_chars=self.prompt_chars,
            client=self.client,
            app_version=self.app_version,
        )

    def _elapsed_ms(self, since: Optional[float]) -> int:
        if since is None:
            return 0
        return max(0, int(round((self._clock() - since) * 1000.0)))

    def _emit(self, event_type: str, event_time: datetime, **fields: Any) -> None:
        try:
            event: Dict[str, Any] = {
                "event_id": new_id(),
                "event_type": event_type,
                "request_id": self.request_id,
                "session_id": self.session_id,
                "user_key": self.user_key,
                "event_time": rfc3339_utc(event_time),
            }
            event.update(fields)
            self._sink.emit(event)
        except Exception:
            logger.debug("Could not emit telemetry event %s", event_type, exc_info=True)


def _add_tokens(total: Optional[int], value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return total
    return value if total is None else total + value


# The telemetry of the agent turn running in this context, if telemetry is on.
current_request: ContextVar[Optional[RequestTelemetry]] = ContextVar(
    "sportsagentic_request_telemetry", default=None
)
