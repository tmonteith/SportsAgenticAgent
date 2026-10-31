"""Pydantic schemas for the FastAPI endpoints."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

ProviderMode = Literal["auto", "llama", "gemini"]


class ConversationMessage(BaseModel):
    """A single message in the conversation history."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None


class ToolCallRecord(BaseModel):
    """Record of an MCP tool call executed during an agent turn."""

    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    result: str


class ChatRequest(BaseModel):
    """Request payload for POST /chat."""

    message: str = Field(..., min_length=1, description="User query to send to the agent")
    history: List[ConversationMessage] = Field(
        default_factory=list,
        description="Optional prior conversation messages",
    )
    provider: Optional[ProviderMode] = Field(
        default=None,
        description="Override inference provider: 'auto', 'llama', or 'gemini'",
    )
    skills: Optional[List[str]] = Field(
        default=None,
        description="Optional list of skill names to activate (defaults to all loaded skills)",
    )


class ChatResponse(BaseModel):
    """Response payload for POST /chat."""

    reply: str
    provider_used: str
    fallback_triggered: bool = False
    tool_calls: List[ToolCallRecord] = Field(default_factory=list)


class ComponentStatus(BaseModel):
    """Health status for an upstream dependency."""

    available: bool
    detail: str


class HealthResponse(BaseModel):
    """Response payload for GET /health."""

    status: Literal["ok", "degraded"]
    mcp_server: ComponentStatus
    llama_server: ComponentStatus
    gemini: ComponentStatus


class ToolInfo(BaseModel):
    """Metadata describing an exposed MCP tool."""

    name: str
    description: str
    parameters: Dict[str, Any] = Field(default_factory=dict)


class ToolsResponse(BaseModel):
    """Response payload for GET /tools."""

    tools: List[ToolInfo]


class SkillInfo(BaseModel):
    """Metadata describing a loaded agent skill."""

    name: str
    description: str
    source_path: str


class SkillsResponse(BaseModel):
    """Response payload for GET /skills."""

    skills: List[SkillInfo]
