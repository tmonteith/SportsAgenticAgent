"""Loader and parser for markdown agent skill files."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from app.logging_config import get_logger

logger = get_logger("skills")

BASE_AGENT_PROMPT = (
    "You are SportsAgenticAgent, a knowledgeable College Football AI assistant connected "
    "to the SportsAgentic MCP server. Use the available MCP tools whenever you need live "
    "scores, team schedules, or national rankings. Answer clearly and accurately based on "
    "the tool output."
)


@dataclass
class SkillDocument:
    """Parsed skill document loaded from a markdown file."""

    name: str
    description: str
    body: str
    source_path: str


def parse_skill_markdown(raw_text: str, default_name: str, source_path: str = "") -> SkillDocument:
    """Parse optional YAML frontmatter and markdown body from a skill file."""
    text = raw_text.strip()
    meta: Dict[str, str] = {}
    body = text

    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            frontmatter_raw = parts[1].strip()
            body = parts[2].strip()
            for line in frontmatter_raw.splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or ":" not in stripped:
                    continue
                key, val = stripped.split(":", 1)
                clean_val = val.strip().strip('"').strip("'")
                meta[key.strip().lower()] = clean_val

    name = meta.get("name") or default_name
    description = meta.get("description") or f"Skill loaded from {source_path or name}"

    return SkillDocument(
        name=name,
        description=description,
        body=body,
        source_path=source_path,
    )


class SkillsLoader:
    """Discovers and loads markdown skill files for the agent."""

    def __init__(
        self,
        skills_dir: str | Path = "skills",
        extra_dirs: Optional[Sequence[str | Path]] = None,
    ) -> None:
        self.skills_dir = Path(skills_dir)
        default_extra = [".agents/skills"] if extra_dirs is None else list(extra_dirs)
        self.extra_dirs: List[Path] = [Path(d) for d in default_extra]
        self._skills: Dict[str, SkillDocument] = {}
        self.reload()

    def reload(self) -> Dict[str, SkillDocument]:
        """Scan skill directories and load all markdown skill definitions."""
        discovered: Dict[str, SkillDocument] = {}
        search_dirs = [self.skills_dir, *self.extra_dirs]

        for directory in search_dirs:
            if not directory.exists() or not directory.is_dir():
                continue

            md_files: List[Path] = sorted(directory.glob("*.md")) + sorted(
                directory.glob("*/SKILL.md")
            )
            for md_path in md_files:
                try:
                    raw = md_path.read_text(encoding="utf-8")
                    default_name = (
                        md_path.parent.name
                        if md_path.name.upper() == "SKILL.MD"
                        else md_path.stem
                    )
                    doc = parse_skill_markdown(
                        raw,
                        default_name=default_name,
                        source_path=str(md_path),
                    )
                    if doc.name not in discovered:
                        discovered[doc.name] = doc
                        logger.debug("Loaded skill '%s' from %s", doc.name, md_path)
                except OSError as exc:
                    logger.warning("Failed to read skill file %s: %s", md_path, exc)

        self._skills = discovered
        logger.info("Loaded %d agent skill(s): %s", len(self._skills), list(self._skills.keys()))
        return self._skills

    def list_skills(self) -> List[SkillDocument]:
        """Return all loaded skills sorted by name."""
        return [self._skills[k] for k in sorted(self._skills.keys())]

    def get_skill(self, name: str) -> Optional[SkillDocument]:
        """Return a specific skill by name."""
        return self._skills.get(name)

    def build_system_prompt(
        self,
        skill_names: Optional[Sequence[str]] = None,
        current_date: Optional[date] = None,
    ) -> str:
        """Assemble the full system prompt including active skill instructions."""
        today_str = (current_date or date.today()).isoformat()
        sections: List[str] = [
            BASE_AGENT_PROMPT,
            f"Current date: {today_str} (Season year: {today_str[:4]}).",
        ]

        selected_docs: List[SkillDocument]
        if skill_names is None:
            selected_docs = self.list_skills()
        else:
            selected_docs = [
                self._skills[name] for name in skill_names if name in self._skills
            ]

        for doc in selected_docs:
            if doc.body:
                sections.append(f"## Skill: {doc.name}\n{doc.body}")

        return "\n\n".join(sections)
