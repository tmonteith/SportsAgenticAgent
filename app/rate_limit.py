"""Per-client sliding-window limit on agent turns."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import time
from typing import Callable, Deque, Dict

from fastapi import Request

# Empty windows are dropped at most this often, so memory tracks recent clients only.
PRUNE_INTERVAL_SECONDS = 60.0


@dataclass(frozen=True)
class RateLimitDecision:
    """Outcome of counting one request against a client's window."""

    allowed: bool
    limit: int
    remaining: int
    retry_after_seconds: int

    def headers(self) -> Dict[str, str]:
        """Response headers describing the client's budget."""
        headers = {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(self.remaining),
        }
        if not self.allowed:
            headers["Retry-After"] = str(self.retry_after_seconds)
        return headers


class SlidingWindowRateLimiter:
    """Allows at most max_requests per client in any window_seconds span.

    State lives in this process, which is right for the single agent replica: counts
    reset when the pod restarts, and more than one replica would need a shared store.
    """

    def __init__(
        self,
        max_requests: int,
        window_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._clock = clock
        self._hits: Dict[str, Deque[float]] = {}
        self._last_prune = clock()

    def hit(self, key: str) -> RateLimitDecision:
        """Count a request for key, unless its window is already full."""
        now = self._clock()
        self._prune(now)

        hits = self._hits.setdefault(key, deque())
        cutoff = now - self.window_seconds
        while hits and hits[0] <= cutoff:
            hits.popleft()

        if len(hits) >= self.max_requests:
            retry_after = max(1, math.ceil(hits[0] + self.window_seconds - now))
            return RateLimitDecision(False, self.max_requests, 0, retry_after)

        hits.append(now)
        return RateLimitDecision(True, self.max_requests, self.max_requests - len(hits), 0)

    def _prune(self, now: float) -> None:
        if now - self._last_prune < PRUNE_INTERVAL_SECONDS:
            return
        self._last_prune = now
        cutoff = now - self.window_seconds
        for key in [k for k, hits in self._hits.items() if not hits or hits[-1] <= cutoff]:
            del self._hits[key]


def client_key(request: Request, ip_header: str) -> str:
    """Identify the caller by the header the ingress sets, or by the socket peer.

    Only trust the header when every request comes through the ingress, which
    overwrites it. ingress-nginx sets X-Real-IP to the client address it saw.
    """
    if ip_header:
        value = request.headers.get(ip_header, "").strip()
        if value:
            return value
    return request.client.host if request.client else "unknown"


def describe_wait(seconds: int) -> str:
    """Render a wait as a short phrase, such as '23 minutes'."""
    if seconds < 60:
        return f"{seconds} second{'s' if seconds != 1 else ''}"
    minutes = math.ceil(seconds / 60)
    return f"{minutes} minute{'s' if minutes != 1 else ''}"


def describe_window(seconds: float) -> str:
    """Render the window as 'hour', '30 minutes', and so on."""
    whole = int(seconds)
    if whole == 3600:
        return "hour"
    if whole % 3600 == 0:
        return f"{whole // 3600} hours"
    if whole % 60 == 0:
        return f"{whole // 60} minutes"
    return f"{whole} seconds"
