"""FastAPI application entry point for SportsAgenticAgent."""

from __future__ import annotations

from contextlib import asynccontextmanager
import time
from typing import AsyncIterator, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app.agent.orchestrator import AgentOrchestrator
from app.config import Settings, get_settings
from app.llm.base import LLMClientError
from app.llm.gemini_client import GeminiClient
from app.llm.llama_client import LlamaServerClient
from app.llm.router import LLMRouter
from app.logging_config import get_logger, setup_logging
from app.mcp.client import SportsAgenticMCPClient
from app.schemas import (
    ChatRequest,
    ChatResponse,
    ComponentStatus,
    HealthResponse,
    SkillInfo,
    SkillsResponse,
    ToolInfo,
    ToolsResponse,
)
from app.skills.loader import SkillsLoader

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

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        logger.info(
            "Starting SportsAgenticAgent (mcp=%s, llama=%s, default_provider=%s)",
            settings.mcp_server_url,
            settings.llama_server_url,
            settings.default_llm_provider,
        )
        yield
        logger.info("Shutting down SportsAgenticAgent")

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

    @application.middleware("http")
    async def log_requests(request: Request, call_next):  # type: ignore[no-untyped-def]
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
        return response

    def get_orchestrator(request: Request) -> AgentOrchestrator:
        return request.app.state.orchestrator

    def get_mcp_client(request: Request) -> SportsAgenticMCPClient:
        return request.app.state.mcp_client

    def get_skills_loader(request: Request) -> SkillsLoader:
        return request.app.state.skills_loader

    @application.post("/chat", response_model=ChatResponse)
    async def chat_endpoint(
        payload: ChatRequest,
        agent: AgentOrchestrator = Depends(get_orchestrator),
    ) -> ChatResponse:
        """Process a user query through the agent orchestrator."""
        try:
            return await agent.run(
                user_message=payload.message,
                history=payload.history,
                provider=payload.provider,
                skill_names=payload.skills,
            )
        except LLMClientError as exc:
            logger.error("Chat request failed due to LLM error: %s", exc)
            raise HTTPException(status_code=502, detail=str(exc)) from exc

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
