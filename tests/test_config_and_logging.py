"""Tests for application configuration and logging setup."""

from __future__ import annotations

import logging
import pytest

from app.config import Settings, get_settings
from app.logging_config import get_logger, setup_logging


def test_settings_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify default configuration values when environment variables are unset."""
    monkeypatch.delenv("MCP_SERVER_URL", raising=False)
    monkeypatch.delenv("LLAMA_SERVER_URL", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("DEFAULT_LLM_PROVIDER", raising=False)

    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.mcp_server_url == "http://localhost:8080/sse"
    assert settings.llama_server_url == "http://localhost:8081/v1"
    assert settings.default_llm_provider == "auto"
    assert settings.max_tool_iterations == 5
    assert settings.port == 8000


def test_settings_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify environment variables override Settings fields."""
    monkeypatch.setenv("MCP_SERVER_URL", "http://127.0.0.1:9090/sse")
    monkeypatch.setenv("LLAMA_SERVER_URL", "http://127.0.0.1:9091/v1")
    monkeypatch.setenv("GEMINI_API_KEY", "secret-gemini-key")
    monkeypatch.setenv("DEFAULT_LLM_PROVIDER", "gemini")
    monkeypatch.setenv("MAX_TOOL_ITERATIONS", "8")

    get_settings.cache_clear()
    settings = get_settings()
    assert settings.mcp_server_url == "http://127.0.0.1:9090/sse"
    assert settings.llama_server_url == "http://127.0.0.1:9091/v1"
    assert settings.gemini_api_key == "secret-gemini-key"
    assert settings.default_llm_provider == "gemini"
    assert settings.max_tool_iterations == 8
    get_settings.cache_clear()


def test_logging_setup_and_child_loggers(caplog: pytest.LogCaptureFixture) -> None:
    """Verify setup_logging configures the sportsagentic logger namespace and emits logs."""
    root_logger = setup_logging("DEBUG")
    assert root_logger.name == "sportsagentic"
    assert root_logger.level == logging.DEBUG

    child = get_logger("mcp")
    assert child.name == "sportsagentic.mcp"

    already_prefixed = get_logger("sportsagentic.llm")
    assert already_prefixed.name == "sportsagentic.llm"

    with caplog.at_level(logging.INFO, logger="sportsagentic"):
        child.info("Connected to MCP server")

    assert "Connected to MCP server" in caplog.text
