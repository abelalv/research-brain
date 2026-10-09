"""`rbrain init` (create a vault) and `rbrain doctor` (diagnose the setup)."""

from __future__ import annotations

import shutil
from importlib import resources
from pathlib import Path

from . import providers
from .config import DEFAULT_CONFIG_TOML, USER_CONFIG, Config, config_path
from .skills import list_skills, packaged_skills_dir
from .vault import TYPE_DIRS, Vault

GITIGNORE = """\
# research-brain runtime state (caches, rebuilt automatically)
.rbrain/embeddings.json
.rbrain/rag/
.rbrain/config.toml.bak
# Obsidian per-device UI state
.obsidian/workspace*.json
.obsidian/cache
.trash/
"""


def _template_dir() -> Path:
    return Path(str(resources.files("research_brain") / "vault_template"))


def _copy(src: Path, dst: Path, force: bool, created: list[Path]) -> None:
    if dst.exists() and not force:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    created.append(dst)


def init_vault(root: Path, force: bool = False, link_skills: bool = False,
               set_default: bool = True) -> list[Path]:
    """Create (or complete) a vault. Existing files are never overwritten unless ``force``."""
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    for folder in TYPE_DIRS.values():
        (root / folder).mkdir(exist_ok=True)
    (root / TYPE_DIRS["source"] / "assets").mkdir(exist_ok=True)

    tpl = _template_dir()
    for src in tpl.rglob("*"):
        if src.is_dir():
            continue
        rel = src.relative_to(tpl)
        if rel.parts[0] == "obsidian":           # shipped without a dot so packaging keeps it
            dst = root / ".obsidian" / Path(*rel.parts[1:])
        elif rel.parts[0] == "copilot-prompts":  # Obsidian Copilot custom prompts
            dst = root / "copilot-custom-prompts" / Path(*rel.parts[1:])
        else:
            dst = root / rel
        _copy(src, dst, force, created)
    # Codex / OpenCode / Gemini read AGENTS.md; keep it identical to CLAUDE.md.
    _copy(root / "CLAUDE.md", root / "AGENTS.md", force, created)

    # Skills: copied (default) so each vault can customise them, or symlinked for development.
    skills_dst = root / ".claude" / "skills"
    skills_dst.mkdir(parents=True, exist_ok=True)
    for skill in packaged_skills_dir().iterdir():
        if not (skill / "SKILL.md").exists():
            continue
        target = skills_dst / skill.name
        if target.exists() or target.is_symlink():
            if not force:
                continue
            shutil.rmtree(target) if target.is_dir() and not target.is_symlink() else target.unlink()
        if link_skills:
            target.symlink_to(skill.resolve(), target_is_directory=True)
        else:
            shutil.copytree(skill, target)
        created.append(target)

    cfg = config_path(root)
    if not cfg.exists() or force:
        cfg.parent.mkdir(parents=True, exist_ok=True)
        cfg.write_text(DEFAULT_CONFIG_TOML, encoding="utf-8")
        created.append(cfg)
    gi = root / ".gitignore"
    if not gi.exists():
        gi.write_text(GITIGNORE, encoding="utf-8")
        created.append(gi)
    if set_default:
        USER_CONFIG.parent.mkdir(parents=True, exist_ok=True)
        USER_CONFIG.write_text(str(root), encoding="utf-8")
    return created


# ------------------------------------------------------------------------- doctor
def doctor(cfg: Config, live: bool = False) -> list[tuple[bool, str]]:
    """Check the vault, every profile in use, embeddings and optional tools.

    With ``live=True`` each profile also answers a one-line prompt (costs a few tokens).
    """
    from .connect import check_profile, test_embeddings  # local import avoids a cycle

    results: list[tuple[bool, str]] = [(True, f"Vault: {cfg.vault}")]
    missing = [d for d in TYPE_DIRS.values() if not (cfg.vault / d).is_dir()]
    results.append((not missing, "Folders OK" if not missing else f"Missing folders: {missing} (run rbrain init)"))
    results.append(((cfg.vault / ".obsidian").is_dir(), "Obsidian config folder (.obsidian)"))
    results.append(((cfg.vault / "CLAUDE.md").exists(), "Schema file CLAUDE.md / AGENTS.md"))

    skills = list_skills(cfg.vault)
    in_vault = (cfg.vault / ".claude" / "skills").is_dir()
    results.append((bool(skills), f"{len(skills)} skills ({'vault copies' if in_vault else 'packaged'}): "
                                  + ", ".join(s.name for s in skills)))

    st = providers.ollama_status()
    if st.running:
        results.append((True, f"Ollama v{st.version} at {st.url} — {len(st.models)} models"
                              + ("" if st.binary else " (server on another host / Windows side)")))
    else:
        results.append((True, "Ollama: " + ("installed but not running" if st.binary else "not detected")
                              + " (only needed for local models)"))

    used = {"default"} | set(cfg.skill_profiles.values())
    seen = set()
    for name, p in cfg.all_profiles().items():
        if (p.provider, p.model) in seen:
            continue
        seen.add((p.provider, p.model))
        r = check_profile(name, p, live=live)
        optional = name not in used
        results.append((r.ok or optional, r.message + (" (optional profile)" if optional and not r.ok else "")))
    for skill, prof in cfg.skill_profiles.items():
        if prof not in cfg.profiles and prof != "default":
            results.append((False, f"skill_profiles: '{skill}' points to undefined profile '{prof}'"))

    if cfg.embeddings.enabled:
        ok, msg = test_embeddings(cfg.embeddings.as_profile())
        results.append((ok, f"Embeddings {cfg.embeddings.provider}:{cfg.embeddings.model} — "
                            + (msg if ok else f"{msg}. Fix: rbrain ollama --pull-missing, or set "
                                              "[embeddings] enabled = false (keyword search only)")))
    else:
        results.append((True, "Embeddings disabled — BM25 keyword search only"))

    try:
        from .llm import make_client
        from .rag import RagIndex
        embedder = make_client(cfg.embeddings.as_profile()) if cfg.embeddings.enabled else None  # no call made
        rag = RagIndex(Vault(cfg.vault), embedder, cfg.embeddings.model if cfg.embeddings.enabled else "",
                       backend=cfg.rag.backend, chunk_chars=cfg.rag.chunk_chars,
                       overlap_chars=cfg.rag.overlap_chars)
        st = rag.stats()
        results.append((True, f"RAG index ({cfg.rag.backend}): {st['passages']} passages from {st['notes']} notes"
                              + ("" if st["passages"] else " — built automatically, or run: rbrain rag index")))
    except Exception as exc:  # e.g. chroma backend selected but chromadb missing
        results.append((False, f"RAG index: {exc}"))

    ok, msg = providers.claude_code_status()
    results.append((True, f"Claude Code (optional, for agent mode and the claude profile): {msg}"))
    return results
