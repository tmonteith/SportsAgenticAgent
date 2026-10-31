"""Shared pytest fixtures for SportsAgenticAgent tests."""

from __future__ import annotations

from pathlib import Path
import pytest

from app.config import Settings


@pytest.fixture()
def sample_skills_dir(tmp_path: Path) -> Path:
    """Create a temporary skills directory with a sample CFB skill."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)
    skill_file = skills_dir / "cfb.md"
    skill_file.write_text(
        "---\n"
        "name: cfb\n"
        "description: College football scores and schedule skill\n"
        "---\n\n"
        "# CFB Instructions\n"
        "Always use get_cfb_scores for score queries.\n",
        encoding="utf-8",
    )
    return skills_dir


@pytest.fixture()
def test_settings(sample_skills_dir: Path) -> Settings:
    """Return isolated Settings for unit/integration tests."""
    return Settings(
        mcp_server_url="http://localhost:8080/sse",
        mcp_timeout_seconds=2.0,
        llama_server_url="http://localhost:8081/v1",
        llama_model="test-llama-model",
        llama_timeout_seconds=2.0,
        gemini_api_key="test-gemini-key",
        gemini_model="gemini-2.5-flash",
        gemini_timeout_seconds=2.0,
        gemini_base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        default_llm_provider="auto",
        max_tool_iterations=3,
        skills_dir=str(sample_skills_dir),
        log_level="DEBUG",
    )
