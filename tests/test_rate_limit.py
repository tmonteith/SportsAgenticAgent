"""Tests for the per-client rate limit on agent turns."""

from __future__ import annotations

from typing import Any, Dict, Optional

import httpx
import pytest
from starlette.requests import Request

from app.config import Settings
from app.main import create_app
from app.rate_limit import (
    PRUNE_INTERVAL_SECONDS,
    SlidingWindowRateLimiter,
    client_key,
    describe_wait,
    describe_window,
)
from app.schemas import ChatResponse


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class FakeAgent:
    """Orchestrator stand-in that answers every turn immediately."""

    def __init__(self) -> None:
        self.turns = 0

    async def run(self, **kwargs: Any) -> ChatResponse:
        self.turns += 1
        return ChatResponse(reply="Georgia is ranked #1.", provider_used="llama")


def make_request(headers: Optional[Dict[str, str]] = None, peer: str = "10.0.10.5") -> Request:
    raw_headers = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    return Request({"type": "http", "headers": raw_headers, "client": (peer, 5000)})


def limited_app(test_settings: Settings, **overrides: Any) -> Any:
    settings = test_settings.model_copy(
        update={"rate_limit_requests": 2, "rate_limit_window_seconds": 3600, **overrides}
    )
    app = create_app(settings)
    app.state.orchestrator = FakeAgent()
    return app


async def post_stream(app: Any, ip: Optional[str] = None, content: str = "Who is #1?") -> httpx.Response:
    headers = {"X-Real-IP": ip} if ip else {}
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        return await client.post(
            "/chat/stream",
            json={"messages": [{"role": "user", "content": content}]},
            headers=headers,
        )


async def post_chat(app: Any, ip: str) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        return await client.post("/chat", json={"message": "Who is #1?"}, headers={"X-Real-IP": ip})


def test_limiter_allows_up_to_the_limit_then_reports_the_wait() -> None:
    clock = FakeClock()
    limiter = SlidingWindowRateLimiter(max_requests=3, window_seconds=3600, clock=clock)

    remaining = []
    for _ in range(3):
        decision = limiter.hit("1.2.3.4")
        assert decision.allowed
        remaining.append(decision.remaining)
        clock.now += 60
    assert remaining == [2, 1, 0]

    denied = limiter.hit("1.2.3.4")
    assert not denied.allowed
    # The oldest hit was 180s ago, so it leaves the window in 3420s.
    assert denied.retry_after_seconds == 3420
    assert denied.headers()["Retry-After"] == "3420"


def test_limiter_window_slides_and_denied_requests_do_not_count() -> None:
    clock = FakeClock()
    limiter = SlidingWindowRateLimiter(max_requests=2, window_seconds=100, clock=clock)

    assert limiter.hit("a").allowed
    clock.now += 50
    assert limiter.hit("a").allowed
    clock.now += 10
    assert not limiter.hit("a").allowed

    clock.now += 41  # first hit is now 101s old
    assert limiter.hit("a").allowed
    assert not limiter.hit("a").allowed


def test_limiter_counts_clients_separately_and_prunes_idle_ones() -> None:
    clock = FakeClock()
    limiter = SlidingWindowRateLimiter(max_requests=1, window_seconds=10, clock=clock)

    assert limiter.hit("a").allowed
    assert limiter.hit("b").allowed
    assert not limiter.hit("a").allowed

    clock.now += PRUNE_INTERVAL_SECONDS + 1
    assert limiter.hit("c").allowed
    assert set(limiter._hits) == {"c"}


def test_client_key_prefers_the_ingress_header() -> None:
    assert client_key(make_request({"X-Real-IP": "203.0.113.7"}), "X-Real-IP") == "203.0.113.7"
    assert client_key(make_request({"X-Real-IP": "  "}), "X-Real-IP") == "10.0.10.5"
    assert client_key(make_request({"X-Real-IP": "203.0.113.7"}), "") == "10.0.10.5"


def test_wait_and_window_descriptions() -> None:
    assert describe_wait(1) == "1 second"
    assert describe_wait(45) == "45 seconds"
    assert describe_wait(61) == "2 minutes"
    assert describe_wait(3420) == "57 minutes"
    assert describe_window(3600) == "hour"
    assert describe_window(7200) == "2 hours"
    assert describe_window(1800) == "30 minutes"
    assert describe_window(45) == "45 seconds"


@pytest.mark.asyncio
async def test_stream_returns_429_after_the_limit(test_settings: Settings) -> None:
    app = limited_app(test_settings)

    first = await post_stream(app, ip="203.0.113.7")
    second = await post_stream(app, ip="203.0.113.7")
    assert first.status_code == 200
    assert first.headers["X-RateLimit-Remaining"] == "1"
    assert second.headers["X-RateLimit-Remaining"] == "0"

    third = await post_stream(app, ip="203.0.113.7")
    assert third.status_code == 429
    assert int(third.headers["Retry-After"]) > 3500
    assert third.json()["detail"] == (
        "Limit of 2 requests per hour reached. Try again in 60 minutes."
    )
    assert app.state.orchestrator.turns == 2

    other_client = await post_stream(app, ip="198.51.100.20")
    assert other_client.status_code == 200


@pytest.mark.asyncio
async def test_chat_shares_the_budget_with_stream(test_settings: Settings) -> None:
    app = limited_app(test_settings)

    assert (await post_chat(app, "203.0.113.7")).status_code == 200
    assert (await post_stream(app, ip="203.0.113.7")).status_code == 200
    limited = await post_chat(app, "203.0.113.7")
    assert limited.status_code == 429
    assert limited.headers["X-RateLimit-Remaining"] == "0"


@pytest.mark.asyncio
async def test_invalid_requests_do_not_use_the_budget(test_settings: Settings) -> None:
    app = limited_app(test_settings)

    for _ in range(3):
        assert (await post_stream(app, ip="203.0.113.7", content="   ")).status_code == 422
    assert (await post_stream(app, ip="203.0.113.7")).status_code == 200


@pytest.mark.asyncio
async def test_rate_limit_can_be_disabled(test_settings: Settings) -> None:
    app = limited_app(test_settings, rate_limit_enabled=False)

    for _ in range(4):
        response = await post_stream(app, ip="203.0.113.7")
        assert response.status_code == 200
        assert "X-RateLimit-Remaining" not in response.headers
