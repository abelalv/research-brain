"""`rbrain mcp`: expose the vault's RAG and tools over the Model Context Protocol (stdio).

With this server registered, Claude Code (agent mode), Claude Desktop or any MCP client can
search the vault passage by passage, read notes and verify arguments as native tools:

    rbrain mcp --install                 # writes <vault>/.mcp.json for Claude Code (project scope)
    claude mcp add research-brain -- rbrain --vault ~/ResearchBrain mcp   # or user scope

Optional dependency: the official MCP Python SDK (``--with mcp`` / ``bash install.sh --with-mcp``).
Nothing is printed to stdout except protocol messages; progress goes to stderr.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from .config import find_vault, load_config
from .logic import FormulaError, check_argument
from .workflows import Brain


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr)


def build_tools(brain: Brain) -> dict:
    """Plain functions (also unit-tested without the MCP SDK)."""

    def search_passages(query: str, k: int = 8, scope: str = "all") -> str:
        """Search the research vault passage by passage (hybrid keyword + semantic retrieval).
        scope: 'all', 'wiki' (the researcher's own notes: atoms, concepts, ideas, arguments,
        summaries) or 'sources' (full text of papers). Returns numbered passages with an
        Obsidian link to cite, e.g. [[note#Heading]]."""
        hits = brain.retrieve(query, k=max(1, min(k, 20)), scope=scope if scope in {"all", "wiki", "sources"} else "all")
        if not hits:
            return "No passages found."
        return "\n\n".join(f"[{h.rank}] {h.chunk.location()} — {h.chunk.note_type}\ncite: {h.chunk.link()}\n"
                           f"{h.chunk.text}" for h in hits)

    def search_notes(query: str, k: int = 10, note_type: str = "") -> str:
        """Find whole notes by topic. note_type filters: source, summary, atom, concept, idea,
        argument, answer (empty = all). Returns slug, type, title and one-line summary."""
        types = {note_type} if note_type else None
        hits = brain.search_index().search(query, k=max(1, min(k, 30)), types=types)
        return "\n".join(f"- [[{n.slug}]] ({n.type}) {n.title} — {n.summary[:160]}" for n, _ in hits) \
            or "No notes found."

    def read_note(name: str) -> str:
        """Read a note in full (Markdown with frontmatter) by slug, title or [[link]]."""
        n = brain.vault.resolve(name)
        return n.text() if n else f"No note matches '{name}'."

    def vault_index() -> str:
        """The vault's catalogue (index.md): every note with a one-line summary, grouped by type."""
        path = brain.vault.root / "index.md"
        return path.read_text(encoding="utf-8") if path.exists() else "index.md not found; run refresh."

    def logic_check(premises: list[str], goal: str) -> str:
        """Decide mechanically (SymPy) whether goal follows from premises in propositional logic.
        Syntax: ~ & | -> <-> and parentheses, e.g. ["Res0 & IC -> Solves", "Res0", "IC"], "Solves".
        Returns VALID / INVALID with a counter-model / VACUOUS."""
        try:
            r = check_argument(premises, goal)
        except FormulaError as exc:
            return f"Formula error: {exc}"
        out = r.verdict
        if r.counterexample:
            out += "\nCounter-model: " + ", ".join(f"{k}={'T' if v else 'F'}" for k, v in sorted(r.counterexample.items()))
        if r.valid and r.essential:
            out += "\nEssential premises: " + ", ".join(premises[i] for i in r.essential)
        return out

    def refresh() -> str:
        """Rebuild index.md and re-index new or edited notes for retrieval. Call after writing notes."""
        brain.vault.update_index()
        return str(brain.refresh_rag(force=True))

    return {f.__name__: f for f in (search_passages, search_notes, read_note, vault_index, logic_check, refresh)}


def serve(vault_arg: str | None) -> None:
    try:  # MCP Python SDK 2.x renamed FastMCP to MCPServer; support both
        from mcp.server.mcpserver import MCPServer as Server  # type: ignore[import-not-found]
    except ImportError:
        try:
            from mcp.server.fastmcp import FastMCP as Server  # type: ignore[import-not-found]
        except ImportError:
            sys.exit("The MCP SDK is not installed: uv tool install --editable . --with mcp "
                     "(or bash install.sh --with-mcp)")
    vault = find_vault(vault_arg)
    brain = Brain(load_config(vault), echo=_stderr)
    server = Server("research-brain")
    for fn in build_tools(brain).values():
        server.tool()(fn)
    _stderr(f"research-brain MCP server on {vault}")
    server.run(transport="stdio")


def install(vault_arg: str | None) -> Path:
    """Write <vault>/.mcp.json so Claude Code (started in the vault) offers the server."""
    vault = find_vault(vault_arg)
    exe = shutil.which("rbrain") or "rbrain"
    path = vault / ".mcp.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data.setdefault("mcpServers", {})["research-brain"] = {
        "command": exe, "args": ["--vault", str(vault), "mcp"]}
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path
