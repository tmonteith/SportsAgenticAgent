"""FastAPI application entry point for SportsAgenticAgent."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import time
from typing import AsyncIterator, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from app.agent.orchestrator import AgentOrchestrator
from app.agent.streaming import stream_agent_turn
from app.config import Settings, get_settings
from app.llm.base import LLMClientError
from app.llm.gemini_client import GeminiClient
from app.llm.llama_client import LlamaServerClient
from app.llm.router import LLMRouter
from app.logging_config import get_logger, request_id_var, setup_logging
from app.mcp.client import SportsAgenticMCPClient
from app.rate_limit import (
    RateLimitDecision,
    SlidingWindowRateLimiter,
    client_key,
    describe_wait,
    describe_window,
)
from app.schemas import (
    ChatRequest,
    ChatResponse,
    ChatStreamRequest,
    ComponentStatus,
    ConversationMessage,
    HealthResponse,
    RateLimitStatusResponse,
    SkillInfo,
    SkillsResponse,
    ToolInfo,
    ToolsResponse,
)
from app.skills.loader import SkillsLoader
from app.telemetry import (
    KafkaTelemetry,
    RequestTelemetry,
    build_telemetry,
    current_request,
    missing_telemetry_settings,
    request_telemetry,
)
from app.telemetry.events import (
    ERROR_CLIENT_DISCONNECTED,
    ERROR_INTERNAL,
    ERROR_LLM,
    REQUEST_ID_HEADER,
    new_id,
)

logger = get_logger("api")


def build_components(settings: Settings) -> tuple[
    SportsAgenticMCPClient,
    LlamaServerClient,
    GeminiClient,
    LLMRouter,
    SkillsLoader,
    AgentOrchestrator,
]:
    """Instantiate core application services from Settings."""
    mcp_client = SportsAgenticMCPClient(
        sse_url=settings.mcp_server_url,
        timeout_seconds=settings.mcp_timeout_seconds,
    )
    llama_client = LlamaServerClient(
        base_url=settings.llama_server_url,
        model=settings.llama_model,
        timeout_seconds=settings.llama_timeout_seconds,
    )
    gemini_client = GeminiClient(
        api_key=settings.gemini_api_key,
        model=settings.gemini_model,
        base_url=settings.gemini_base_url,
        timeout_seconds=settings.gemini_timeout_seconds,
    )
    llm_router = LLMRouter(
        llama_client=llama_client,
        gemini_client=gemini_client,
        default_provider=settings.default_llm_provider,
    )
    skills_loader = SkillsLoader(skills_dir=settings.skills_dir)
    orchestrator = AgentOrchestrator(
        mcp_client=mcp_client,
        llm_router=llm_router,
        skills_loader=skills_loader,
        max_tool_iterations=settings.max_tool_iterations,
        tool_result_max_chars=settings.tool_result_max_chars,
    )
    return mcp_client, llama_client, gemini_client, llm_router, skills_loader, orchestrator


def create_app(custom_settings: Optional[Settings] = None) -> FastAPI:
    """Create and configure the FastAPI application instance."""
    settings = custom_settings or get_settings()
    setup_logging(settings.log_level)

    (
        mcp_client,
        llama_client,
        gemini_client,
        llm_router,
        skills_loader,
        orchestrator,
    ) = build_components(settings)
    telemetry = build_telemetry(settings)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        logger.info(
            "Starting SportsAgenticAgent (mcp=%s, llama=%s, default_provider=%s)",
            settings.mcp_server_url,
            settings.llama_server_url,
            settings.default_llm_provider,
        )
        app_telemetry: Optional[KafkaTelemetry] = application.state.telemetry
        if app_telemetry is None:
            missing = missing_telemetry_settings(settings)
            logger.info(
                "Telemetry off: %s not set, so no agent.events are produced",
                " and ".join(missing) if missing else "no telemetry sink",
            )
        else:
            # Returns at once; the producer connects in the background.
            await app_telemetry.start()
        try:
            yield
        finally:
            logger.info("Shutting down SportsAgenticAgent")
            if app_telemetry is not None:
                await app_telemetry.stop()

    application = FastAPI(
        title="SportsAgenticAgent",
        description="FastAPI College Football Agent backed by SportsAgentic MCP, llama-server, and Gemini failover",
        version="0.1.0",
        lifespan=lifespan,
    )

    application.state.settings = settings
    application.state.mcp_client = mcp_client
    application.state.llama_client = llama_client
    application.state.gemini_client = gemini_client
    application.state.llm_router = llm_router
    application.state.skills_loader = skills_loader
    application.state.orchestrator = orchestrator
    application.state.telemetry = telemetry
    application.state.rate_limiter = (
        SlidingWindowRateLimiter(
            max_requests=settings.rate_limit_requests,
            window_seconds=settings.rate_limit_window_seconds,
        )
        if settings.rate_limit_enabled
        else None
    )

    @application.middleware("http")
    async def log_requests(request: Request, call_next):  # type: ignore[no-untyped-def]
        # Always our own UUID v4: a client-supplied id is never trusted or reused.
        request_id = new_id()
        request.state.request_id = request_id
        token = request_id_var.set(request_id)
        try:
            start = time.monotonic()
            response = await call_next(request)
            duration_ms = (time.monotonic() - start) * 1000.0
            logger.info(
                "%s %s -> %d (%.1fms)",
                request.method,
                request.url.path,
                response.status_code,
                duration_ms,
            )
            response.headers[REQUEST_ID_HEADER] = request_id
            return response
        finally:
            request_id_var.reset(token)

    def enforce_rate_limit(request: Request) -> Optional[RateLimitDecision]:
        """Count an agent turn for the caller, raising 429 once their window is full.

        Called from the endpoint body rather than as a dependency, so requests that
        fail validation are rejected without using up the caller's budget.
        """
        limiter: Optional[SlidingWindowRateLimiter] = request.app.state.rate_limiter
        if limiter is None:
            return None
        key = client_key(request, settings.rate_limit_client_ip_header)
        decision = limiter.hit(key)
        if not decision.allowed:
            logger.warning(
                "Rate limit reached for client %s (%d per %ss)",
                key,
                decision.limit,
                settings.rate_limit_window_seconds,
            )
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Limit of {decision.limit} requests per "
                    f"{describe_window(settings.rate_limit_window_seconds)} reached. "
                    f"Try again in {describe_wait(decision.retry_after_seconds)}."
                ),
                headers=decision.headers(),
            )
        return decision

    def turn_telemetry(request: Request, prompt: str) -> Optional[RequestTelemetry]:
        """Telemetry for an agent turn that passed the rate limit, or None when it is off."""
        return request_telemetry(
            request,
            settings,
            request.app.state.telemetry,
            getattr(request.state, "request_id", None) or new_id(),
            prompt,
        )

    def get_orchestrator(request: Request) -> AgentOrchestrator:
        return request.app.state.orchestrator

    def get_mcp_client(request: Request) -> SportsAgenticMCPClient:
        return request.app.state.mcp_client

    def get_skills_loader(request: Request) -> SkillsLoader:
        return request.app.state.skills_loader

    @application.post("/chat", response_model=ChatResponse)
    async def chat_endpoint(
        payload: ChatRequest,
        request: Request,
        response: Response,
        agent: AgentOrchestrator = Depends(get_orchestrator),
    ) -> ChatResponse:
        """Process a user query through the agent orchestrator."""
        decision = enforce_rate_limit(request)
        if decision:
            response.headers.update(decision.headers())
        telemetry = turn_telemetry(request, payload.message)
        if telemetry is not None:
            telemetry.start()
        token = current_request.set(telemetry)
        try:
            result = await agent.run(
                user_message=payload.message,
                history=payload.history,
                provider=payload.provider,
                skill_names=payload.skills,
            )
        except LLMClientError as exc:
            if telemetry is not None:
                telemetry.fail(ERROR_LLM)
            logger.error("Chat request failed due to LLM error: %s", exc)
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except BaseException as exc:
            if telemetry is not None:
                cancelled = isinstance(exc, asyncio.CancelledError)
                telemetry.fail(ERROR_CLIENT_DISCONNECTED if cancelled else ERROR_INTERNAL)
            raise
        finally:
            current_request.reset(token)
        if telemetry is not None:
            telemetry.complete()
        return result

    @application.post("/chat/stream")
    async def chat_stream_endpoint(
        payload: ChatStreamRequest,
        request: Request,
        agent: AgentOrchestrator = Depends(get_orchestrator),
    ) -> StreamingResponse:
        """Stream an agent turn as server-sent events."""
        decision = enforce_rate_limit(request)
        headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        if decision:
            headers.update(decision.headers())
        history = [
            ConversationMessage(role=item.role, content=item.content)
            for item in payload.messages[:-1]
        ]
        user_message = payload.messages[-1].content
        return StreamingResponse(
            stream_agent_turn(
                agent,
                user_message=user_message,
                history=history,
                telemetry=turn_telemetry(request, user_message),
            ),
            media_type="text/event-stream",
            headers=headers,
        )

    @application.get("/rate-limit", response_model=RateLimitStatusResponse)
    async def rate_limit_endpoint(request: Request) -> RateLimitStatusResponse:
        """Report the caller's remaining agent turns without using one."""
        limiter: Optional[SlidingWindowRateLimiter] = request.app.state.rate_limiter
        if limiter is None:
            return RateLimitStatusResponse(enabled=False)
        status = limiter.status(client_key(request, settings.rate_limit_client_ip_header))
        return RateLimitStatusResponse(
            enabled=True,
            limit=status.limit,
            remaining=status.remaining,
            window_seconds=settings.rate_limit_window_seconds,
            next_seconds=status.next_seconds,
            reset_seconds=status.reset_seconds,
        )

    @application.get("/health", response_model=HealthResponse)
    async def health_endpoint(request: Request) -> HealthResponse:
        """Report health of SportsAgentic MCP, llama-server, and Gemini backup."""
        mcp_c: SportsAgenticMCPClient = request.app.state.mcp_client
        llama_c: LlamaServerClient = request.app.state.llama_client
        gemini_c: GeminiClient = request.app.state.gemini_client

        mcp_ok, mcp_detail = await mcp_c.check_health()
        llama_ok, llama_detail = await llama_c.check_health()
        gemini_ok, gemini_detail = await gemini_c.check_health()

        inference_ok = llama_ok or gemini_ok
        overall = "ok" if (mcp_ok and inference_ok) else "degraded"

        return HealthResponse(
            status=overall,
            mcp_server=ComponentStatus(available=mcp_ok, detail=mcp_detail),
            llama_server=ComponentStatus(available=llama_ok, detail=llama_detail),
            gemini=ComponentStatus(available=gemini_ok, detail=gemini_detail),
        )

    @application.get("/tools", response_model=ToolsResponse)
    async def tools_endpoint(
        mcp_c: SportsAgenticMCPClient = Depends(get_mcp_client),
    ) -> ToolsResponse:
        """List MCP tools exposed by SportsAgentic."""
        raw_tools = await mcp_c.list_tools(fallback_to_defaults=True)
        tools = [
            ToolInfo(
                name=str(t.get("name", "")),
                description=str(t.get("description", "")),
                parameters=t.get("inputSchema") or t.get("parameters") or {},
            )
            for t in raw_tools
            if t.get("name")
        ]
        return ToolsResponse(tools=tools)

    @application.get("/skills", response_model=SkillsResponse)
    async def skills_endpoint(
        loader: SkillsLoader = Depends(get_skills_loader),
    ) -> SkillsResponse:
        """List loaded agent skill files."""
        skills = [
            SkillInfo(
                name=doc.name,
                description=doc.description,
                source_path=doc.source_path,
            )
            for doc in loader.list_skills()
        ]
        return SkillsResponse(skills=skills)

    return application


app = create_app()
