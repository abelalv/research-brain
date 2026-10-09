"""Load Agent Skills (SKILL.md) and turn them into prompts for any LLM.

The same skill folder serves two runtimes:

* **Claude Code** discovers ``<vault>/.claude/skills/<name>/SKILL.md`` and follows the whole
  file, including the "In Claude Code" section that tells it how to write notes itself.
* **The rbrain CLI** (local models, DeepSeek, or Claude through the API) reads only the
  model-facing sections (Purpose, Method, Quality checklist), appends the JSON contract
  from ``schema.json``, and lets Python render the Obsidian notes deterministically.

Small local models are much more reliable when they only have to fill a JSON object and
the code takes care of frontmatter, file names, wikilinks, the index and the log.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from .vault import parse_note

MODEL_SECTIONS = ("Purpose", "Method", "Quality checklist")


@dataclass
class Skill:
    name: str
    description: str
    sections: dict[str, str]
    schema: dict | None
    path: Path

    def extra_schema(self, stage: str) -> dict:
        """Schema for a secondary stage, e.g. ``revise.schema.json`` next to SKILL.md."""
        return json.loads((self.path.parent / f"{stage}.schema.json").read_text(encoding="utf-8"))

    def system_prompt(self, language: str = "English", extra: str = "",
                      schema: dict | None = None) -> str:
        schema = schema if schema is not None else self.schema
        parts = [
            "You are the research assistant that maintains a researcher's second brain: an "
            "Obsidian vault of interlinked Markdown notes. You are rigorous, concise and honest "
            "about uncertainty. Never invent sources, citations or results.",
            f"# Skill: {self.name}",
        ]
        for title in MODEL_SECTIONS:
            if title in self.sections:
                parts.append(f"## {title}\n{self.sections[title].strip()}")
        parts.append(
            f"## Language and notation\nWrite all content in {language}. Use LaTeX for "
            "mathematics: inline $...$ and display $$...$$."
        )
        if schema is not None:
            parts.append(
                "## Output format\nReturn ONLY one JSON object (no prose, no code fences) that "
                "follows this schema. Inside JSON strings, write LaTeX backslashes doubled, "
                "e.g. \"$\\\\frac{a}{b}$\".\n" + json.dumps(schema, indent=2, ensure_ascii=False)
            )
        if extra:
            parts.append(extra)
        return "\n\n".join(parts)


def _split_sections(body: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    current, buf = None, []
    for line in body.splitlines():
        if line.startswith("## "):
            if current:
                sections[current] = "\n".join(buf).strip()
            current, buf = line[3:].strip(), []
        elif current:
            buf.append(line)
    if current:
        sections[current] = "\n".join(buf).strip()
    return sections


def packaged_skills_dir() -> Path:
    return Path(str(resources.files("research_brain") / "skills"))


def skill_dirs(vault_root: Path | None) -> list[Path]:
    """Vault copies win over packaged defaults, so you can customise skills per vault."""
    dirs = []
    if vault_root is not None:
        dirs.append(vault_root / ".claude" / "skills")
    dirs.append(packaged_skills_dir())
    return dirs


def load_skill(name: str, vault_root: Path | None = None) -> Skill:
    for base in skill_dirs(vault_root):
        path = base / name / "SKILL.md"
        if path.exists():
            meta, body = parse_note(path.read_text(encoding="utf-8"))
            schema_path = path.parent / "schema.json"
            schema = json.loads(schema_path.read_text(encoding="utf-8")) if schema_path.exists() else None
            return Skill(
                name=meta.get("name", name),
                description=meta.get("description", ""),
                sections=_split_sections(body),
                schema=schema,
                path=path,
            )
    raise FileNotFoundError(f"Skill '{name}' not found in {', '.join(map(str, skill_dirs(vault_root)))}")


def list_skills(vault_root: Path | None = None) -> list[Skill]:
    seen: dict[str, Skill] = {}
    for base in skill_dirs(vault_root):
        if base.exists():
            for skill_md in sorted(base.glob("*/SKILL.md")):
                name = skill_md.parent.name
                if name not in seen:
                    seen[name] = load_skill(name, vault_root)
    return list(seen.values())
