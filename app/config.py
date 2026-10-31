"""Application configuration backed by environment variables and .env files."""

from functools import lru_cache
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for SportsAgenticAgent."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # SportsAgentic MCP Server
    mcp_server_url: str = Field(
        default="http://localhost:8080/sse",
        description="SSE endpoint of the SportsAgentic MCP server",
    )
    mcp_timeout_seconds: float = Field(
        default=10.0,
        gt=0.0,
        description="Timeout in seconds for MCP SSE handshake and tool calls",
    )

    # llama-server (Primary Inference)
    llama_server_url: str = Field(
        default="http://localhost:8081/v1",
        description="OpenAI-compatible base URL for llama-server",
    )
    llama_model: str = Field(
        default="local-model",
        description="Model identifier passed to llama-server",
    )
    llama_timeout_seconds: float = Field(
        default=15.0,
        gt=0.0,
        description="Timeout in seconds before llama-server requests fail over to Gemini",
    )

    # Google Gemini (Backup / Alternative Inference)
    gemini_api_key: Optional[str] = Field(
        default=None,
        description="Google Gemini API key used when failing over or selecting Gemini directly",
    )
    gemini_model: str = Field(
        default="gemini-2.5-flash",
        description="Gemini model name",
    )
    gemini_timeout_seconds: float = Field(
        default=30.0,
        gt=0.0,
        description="Timeout in seconds for Gemini API requests",
    )
    gemini_base_url: str = Field(
        default="https://generativelanguage.googleapis.com/v1beta/openai",
        description="Gemini OpenAI-compatible base URL",
    )

    # Agent & App Runtime
    default_llm_provider: Literal["auto", "llama", "gemini"] = Field(
        default="auto",
        description="Default inference routing mode: auto (llama primary + gemini backup), llama, or gemini",
    )
    max_tool_iterations: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum MCP tool-calling iterations per user request",
    )
    skills_dir: str = Field(
        default="skills",
        description="Directory containing markdown skill files for the agent",
    )
    log_level: str = Field(
        default="INFO",
        description="Root/application logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="Port for the FastAPI application",
    )


@lru_cache()
def get_settings() -> Settings:
    """Return cached application settings instance."""
    return Settings()
