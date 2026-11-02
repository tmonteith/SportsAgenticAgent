"""Unit tests for the agent skills markdown loader."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional, Tuple

import pytest

from app.skills.loader import SkillsLoader, current_cfb_weeks, parse_skill_markdown


def test_parse_skill_markdown_with_frontmatter() -> None:
    """Verify YAML frontmatter and markdown body are parsed accurately."""
    raw = (
        "---\n"
        "name: cfb\n"
        "description: Query college football scores and rankings.\n"
        "---\n\n"
        "# College Football Skill\n"
        "Use `get_cfb_scores` for live games.\n"
    )
    doc = parse_skill_markdown(raw, default_name="fallback", source_path="skills/cfb.md")
    assert doc.name == "cfb"
    assert doc.description == "Query college football scores and rankings."
    assert "# College Football Skill" in doc.body
    assert doc.source_path == "skills/cfb.md"


def test_skills_loader_discovers_and_builds_prompt(sample_skills_dir: Path) -> None:
    """Verify SkillsLoader loads skills from directory and injects them into system prompt."""
    loader = SkillsLoader(skills_dir=sample_skills_dir, extra_dirs=[])
    skills = loader.list_skills()
    assert len(skills) == 1
    assert skills[0].name == "cfb"
    assert loader.get_skill("cfb") is not None
    assert loader.get_skill("nonexistent") is None

    prompt = loader.build_system_prompt(current_date=date(2026, 10, 17))
    assert "Current date: 2026-10-17" in prompt
    assert "## Skill: cfb" in prompt
    assert "Always use get_cfb_scores for score queries." in prompt

    # Filtered to empty skill list
    minimal_prompt = loader.build_system_prompt(skill_names=[], current_date=date(2026, 10, 17))
    assert "## Skill: cfb" not in minimal_prompt


@pytest.mark.parametrize(
    ("today", "expected"),
    [
        (date(2026, 8, 31), None),  # Before week 1
        (date(2026, 9, 1), (1, 1)),  # Week 1 opens; the preseason poll is labelled week 1
        (date(2026, 9, 5), (1, 1)),  # First Saturday of the 2026 season
        (date(2026, 9, 6), (1, 2)),  # Sunday poll is labelled with the upcoming week
        (date(2026, 10, 31), (9, 9)),  # Saturday of week 9
        (date(2026, 11, 1), (9, 10)),
        (date(2026, 11, 2), (9, 10)),
        (date(2026, 11, 3), (10, 10)),
        (date(2025, 8, 30), (1, 1)),  # 2025's Labor Day was September 1
        (date(2026, 12, 14), (15, 16)),
        (date(2026, 12, 15), None),  # Postseason
        (date(2027, 1, 10), None),
    ],
)
def test_current_cfb_weeks(today: date, expected: Optional[Tuple[int, int]]) -> None:
    """Verify week numbers match the labels CFBD uses for games and polls."""
    assert current_cfb_weeks(today) == expected


def test_system_prompt_states_weeks_only_during_the_season(sample_skills_dir: Path) -> None:
    """Verify the model is given week numbers in season and none outside it."""
    loader = SkillsLoader(skills_dir=sample_skills_dir, extra_dirs=[])

    in_season = loader.build_system_prompt(current_date=date(2026, 11, 1))
    assert "Current game week: 9. Latest rankings week: 10." in in_season

    offseason = loader.build_system_prompt(current_date=date(2027, 3, 1))
    assert "game week" not in offseason


def test_repository_skill_files_exist_and_load() -> None:
    """Verify the repository's actual skills/cfb.md and .agents/skills/sports-agent/SKILL.md load cleanly."""
    loader = SkillsLoader(skills_dir="skills", extra_dirs=[".agents/skills"])
    skill_names = {s.name for s in loader.list_skills()}
    assert "cfb" in skill_names
    assert "sports-agent" in skill_names
