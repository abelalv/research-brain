"""Reading and writing the Obsidian vault.

The vault is plain Markdown with YAML frontmatter, organised in numbered folders so that
Obsidian's file explorer shows the knowledge pipeline in order:

    00-inbox  -> 10-sources -> 20-summaries -> 30-atoms -> 40-concepts
              -> 50-ideas   -> 60-arguments -> 70-answers

Two special files follow Karpathy's LLM-wiki pattern: ``index.md`` (content catalogue,
regenerated deterministically) and ``log.md`` (append-only, grep-able timeline).
"""

from __future__ import annotations

import datetime as dt
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import yaml

TYPE_DIRS: dict[str, str] = {
    "inbox": "00-inbox",
    "source": "10-sources",
    "summary": "20-summaries",
    "atom": "30-atoms",
    "concept": "40-concepts",
    "idea": "50-ideas",
    "argument": "60-arguments",
    "answer": "70-answers",
    "template": "90-templates",
}
INDEX_ORDER = ["summary", "concept", "idea", "argument", "atom", "answer", "source"]
PLURALS = {"summary": "summaries", "concept": "concepts", "idea": "ideas", "argument": "arguments",
           "atom": "atoms", "answer": "answers", "source": "sources"}
INDEX_TITLES = {
    "summary": "Source summaries", "concept": "Concepts", "idea": "Ideas and hypotheses",
    "argument": "Deductive arguments", "atom": "Atomic ideas", "answer": "Filed answers",
    "source": "Raw sources",
}
SPECIAL_FILES = {"index.md", "log.md", "CLAUDE.md", "AGENTS.md", "Dashboard.md", "lint-report.md"}
EXCLUDED_DIRS = {TYPE_DIRS["template"], "copilot-custom-prompts"}
WIKILINK_RE = re.compile(r"\[\[([^\]|#^]+)(?:[#^][^\]|]*)?(?:\|[^\]]*)?\]\]")
HUMAN_SECTION = "## My notes"


def now_iso() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M")


def today() -> str:
    return dt.date.today().isoformat()


def _as_date(value) -> dt.date:
    """Frontmatter dates are written as YAML dates (unquoted) so Dataview sorts them."""
    if isinstance(value, dt.date):
        return value
    try:
        return dt.date.fromisoformat(str(value)[:10])
    except ValueError:
        return dt.date.today()


def truncate(text: str, limit: int) -> str:
    """Shorten text without leaving an unbalanced $ (which would break math rendering)."""
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    cut = text[:limit]
    if cut.count("$") % 2 == 1:
        cut = cut[:cut.rfind("$")]
    return cut.rstrip() + "…"


def slugify(text: str, max_len: int = 80) -> str:
    """'Physics-Informed Neural Networks (PINNs)' -> 'physics-informed-neural-networks-pinns'.

    Accents are folded (ecuación -> ecuacion) so file names stay portable.
    """
    norm = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    norm = re.sub(r"[^a-zA-Z0-9]+", "-", norm.lower()).strip("-")
    if len(norm) > max_len:  # cut at a word boundary
        cut = norm[:max_len]
        norm = cut[:cut.rfind("-")] if "-" in cut else cut
    return norm.strip("-") or "untitled"


def parse_note(text: str) -> tuple[dict, str]:
    """Split a Markdown file into (frontmatter dict, body)."""
    if text.startswith("---"):
        parts = text.split("\n---", 1)
        if len(parts) == 2:
            header = parts[0][3:]
            body = parts[1].lstrip("-").lstrip("\n")
            try:
                meta = yaml.safe_load(header) or {}
                if isinstance(meta, dict):
                    return meta, body
            except yaml.YAMLError:
                pass
    return {}, text


def dump_note(meta: dict, body: str) -> str:
    header = yaml.safe_dump(meta, sort_keys=False, allow_unicode=True, width=1000).strip()
    return f"---\n{header}\n---\n\n{body.strip()}\n"


def wikilinks(text: str) -> set[str]:
    """Return the link targets in a note (``[[target|alias]]`` -> ``target``)."""
    return {m.strip() for m in WIKILINK_RE.findall(text)}


def link(slug: str, title: str | None = None) -> str:
    return f"[[{slug}|{title}]]" if title and title != slug else f"[[{slug}]]"


@dataclass
class Note:
    path: Path
    meta: dict = field(default_factory=dict)
    body: str = ""

    @property
    def slug(self) -> str:
        return self.path.stem

    @property
    def title(self) -> str:
        return str(self.meta.get("title") or self.slug)

    @property
    def type(self) -> str:
        return str(self.meta.get("type", ""))

    @property
    def summary(self) -> str:
        if self.meta.get("summary"):
            return str(self.meta["summary"])
        for line in self.body.splitlines():
            line = line.strip()
            if line and not line.startswith(("#", ">", "|", "-", "!", "```")):
                return line[:200]
        return ""

    def text(self) -> str:
        return dump_note(self.meta, self.body)

    def link(self) -> str:
        return link(self.slug, self.title)


class Vault:
    def __init__(self, root: Path):
        self.root = Path(root)

    # ------------------------------------------------------------------ paths
    def dir_for(self, note_type: str) -> Path:
        return self.root / TYPE_DIRS[note_type]

    def unique_path(self, note_type: str, title: str, ext: str = ".md",
                    reserved: set[str] | None = None) -> Path:
        """A path whose stem is unique across the whole vault (so [[slug]] is unambiguous).

        ``reserved`` holds slugs already promised to notes that are not written yet; the
        chosen slug is added to it.
        """
        folder = self.dir_for(note_type)
        folder.mkdir(parents=True, exist_ok=True)
        base = slugify(title)
        existing = {p.stem for p in self.root.rglob("*.md")} | (reserved or set())
        candidate, k = base, 2
        while candidate in existing or (folder / f"{candidate}{ext}").exists():
            candidate = f"{base}-{k}"
            k += 1
        if reserved is not None:
            reserved.add(candidate)
        return folder / f"{candidate}{ext}"

    # ------------------------------------------------------------------ reading
    def iter_notes(self, types: Iterable[str] | None = None) -> Iterator[Note]:
        wanted = set(types) if types else None
        for path in sorted(self.root.rglob("*.md")):
            rel = path.relative_to(self.root)
            if any(part.startswith(".") for part in rel.parts):
                continue  # .obsidian, .claude, .rbrain
            if path.name in SPECIAL_FILES or rel.parts[0] in EXCLUDED_DIRS:
                continue
            meta, body = parse_note(path.read_text(encoding="utf-8"))
            note = Note(path, meta, body)
            if wanted is None or note.type in wanted:
                yield note

    def read(self, path: Path) -> Note:
        meta, body = parse_note(path.read_text(encoding="utf-8"))
        return Note(path, meta, body)

    def resolve(self, ref: str) -> Note | None:
        """Find a note from a path, slug, title or ``[[wikilink]]``."""
        ref = ref.strip()
        m = WIKILINK_RE.fullmatch(ref)
        if m:
            ref = m.group(1)
        candidate = Path(ref).expanduser()
        for p in (candidate, self.root / candidate):
            if p.suffix == ".md" and p.exists():
                return self.read(p)
        target = slugify(Path(ref).stem)
        title_hit = None
        for note in self.iter_notes():
            if note.slug == ref or note.slug == target:
                return note
            if title_hit is None and slugify(note.title) == target:
                title_hit = note
        return title_hit

    # ------------------------------------------------------------------ writing
    def write(self, note_type: str, title: str, meta: dict, body: str,
              path: Path | None = None) -> Note:
        """Write a note. If ``path`` exists, a human-owned "## My notes" section is preserved."""
        path = path or self.unique_path(note_type, title)
        path.parent.mkdir(parents=True, exist_ok=True)
        created = dt.date.today()
        if path.exists():
            old_meta, old_body = parse_note(path.read_text(encoding="utf-8"))
            created = _as_date(old_meta.get("created", created))
            if HUMAN_SECTION in old_body and HUMAN_SECTION not in body:
                body = body.rstrip() + "\n\n" + old_body[old_body.index(HUMAN_SECTION):]
        full_meta = {"title": title, "type": note_type,
                     **{k: v for k, v in meta.items() if v not in (None, "", [])}}
        full_meta["created"] = _as_date(full_meta.get("created", created))
        full_meta["updated"] = dt.date.today()
        note = Note(path, full_meta, body)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(note.text(), encoding="utf-8")
        tmp.replace(path)  # atomic on POSIX
        return note

    def append_log(self, operation: str, title: str, created: Iterable[Note] = ()) -> None:
        """Append a Karpathy-style entry: ``## [YYYY-MM-DD HH:MM] op | title``."""
        log = self.root / "log.md"
        if not log.exists():
            log.write_text("# Log\n\nAppend-only timeline of operations on this vault.\n", encoding="utf-8")
        lines = [f"\n## [{now_iso()}] {operation} | {title}"]
        lines += [f"- {n.link()}" for n in created]
        with log.open("a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")

    def update_index(self) -> Path:
        """Regenerate ``index.md``: one line per note, grouped by type."""
        groups: dict[str, list[Note]] = {t: [] for t in INDEX_ORDER}
        for note in self.iter_notes():
            if note.type in groups:
                groups[note.type].append(note)
        counts = ", ".join(f"{len(groups[t])} {t if len(groups[t]) == 1 else PLURALS[t]}"
                           for t in INDEX_ORDER if groups[t])
        out = [
            "# Index",
            "",
            f"_Regenerated by `rbrain index` on {now_iso()}. {counts or 'Empty vault.'}_",
            "",
            "Read this file first to find relevant pages, then open them.",
        ]
        for t in INDEX_ORDER:
            if not groups[t]:
                continue
            out += ["", f"## {INDEX_TITLES[t]}", ""]
            for note in sorted(groups[t], key=lambda n: n.title.lower()):
                summary = truncate(note.summary, 140)
                out.append(f"- {note.link()}" + (f" — {summary}" if summary else ""))
        path = self.root / "index.md"
        path.write_text("\n".join(out) + "\n", encoding="utf-8")
        return path
