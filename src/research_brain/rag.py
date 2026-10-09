"""Passage-level retrieval-augmented generation (RAG) over the whole vault.

Karpathy's LLM-wiki pattern compiles knowledge into notes so it is not re-derived on every
question. RAG complements it: the wiki holds the *synthesis* (atoms, concepts, ideas,
arguments), the raw sources hold the *evidence* (exact numbers, definitions, quotes). This
module indexes both at passage level so an answer can cite the precise section or PDF page.

Pipeline
--------
1. **Chunking** (``chunk_note``): heading-aware and page-aware windows of ~``chunk_chars``
   characters with overlap; code blocks are never split; each chunk keeps its note, heading
   path and PDF page.
2. **Contextual headers**: every chunk is embedded and keyword-indexed together with
   "note title › section" so short passages remain retrievable (a cheap, LLM-free variant of
   contextual retrieval).
3. **Incremental index** (``RagIndex.update``): notes are hashed; only new or edited notes are
   re-chunked and re-embedded; deleted notes are removed. Vectors live in NumPy files (no
   extra dependency) or in ChromaDB (optional).
4. **Hybrid retrieval** (``RagIndex.search``): BM25 on chunks + dense cosine similarity, for
   the question and optional rewritten variants, fused with Reciprocal Rank Fusion; scope
   filters (wiki / sources / all); MMR-style diversification and a per-note cap.
5. **Context assembly** (``build_context``): passages are numbered [S1], [S2], ... and clipped
   (never silently dropped) to a character budget; the report says exactly what was sent.
6. **Grounded citations** (``link_citations``): [S#] markers in the answer become clickable
   Obsidian links ``[[note#Heading|S1]]``; invalid markers are flagged; a groundedness score
   reports the share of sentences that carry a citation.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .llm import LLMClient, LLMError
from .search import BM25, tokenize
from .vault import Note, Vault

WIKI_TYPES = {"summary", "atom", "concept", "idea", "argument", "answer"}
SCOPES = {"all": None, "wiki": WIKI_TYPES, "sources": {"source"}, "inbox": {"inbox"}}
INDEX_VERSION = 2

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_PAGE_HEADING_RE = re.compile(r"^page\s+(\d+)$", re.I)
_PAGE_MARK_RE = re.compile(r"^\[page\s+(\d+)\]$", re.I)
_IMMUTABLE_RE = re.compile(r"> \[!warning\] Immutable source\n(?:>.*\n?)*")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z$\\(\[])")


# =========================================================================== chunks
@dataclass
class Chunk:
    id: str
    slug: str
    title: str
    note_type: str
    heading: str          # "Methods › Loss function" (level-1 title excluded)
    page: int | None
    text: str

    @property
    def header(self) -> str:
        return self.title + (f" › {self.heading}" if self.heading else "")

    def embed_text(self, prefix: str = "") -> str:
        return f"{prefix}{self.header}\n{self.text}"

    def link(self, alias: str | None = None) -> str:
        """Obsidian link to the note, and to its deepest heading when it is link-safe."""
        target = self.slug
        last = self.heading.split(" › ")[-1] if self.heading else ""
        if last and not re.search(r"[\[\]|#^]", last):
            target += f"#{last}"
        elif self.page is not None:
            target += f"#Page {self.page}"
        return f"[[{target}|{alias}]]" if alias else f"[[{target}]]"

    def location(self) -> str:
        loc = self.header
        return loc + (f" (p. {self.page})" if self.page is not None else "")


def _tail(window: str, overlap: int) -> str:
    """Overlap = the last whole paragraphs (or sentences) of the previous window, never a
    fragment of a formula or a word."""
    if overlap <= 0:
        return ""
    paras = window.split("\n\n")
    tail: list[str] = []
    total = 0
    for p in reversed(paras):
        if total + len(p) > overlap:
            break
        tail.insert(0, p)
        total += len(p)
    if tail:
        return "\n\n".join(tail)
    if "$$" in paras[-1] or paras[-1].lstrip().startswith("```"):
        return ""
    sents: list[str] = []
    total = 0
    for sent in reversed(_SENTENCE_SPLIT.split(paras[-1])):
        if total + len(sent) > overlap:
            break
        sents.insert(0, sent)
        total += len(sent)
    text = " ".join(sents)
    return "" if text.count("$") % 2 else text


def _atomic(para: str) -> bool:
    return para.lstrip().startswith("```") or "$$" in para


def _windows(paragraphs: list[str], size: int, overlap: int) -> list[str]:
    """Pack paragraphs into windows of ~size chars; long prose paragraphs are split by
    sentences, while code blocks and display equations are kept whole."""
    pieces: list[str] = []
    for para in paragraphs:
        if len(para) <= size or (_atomic(para) and len(para) <= max(3 * size, 4000)):
            pieces.append(para)
            continue
        buf = ""
        for sent in _SENTENCE_SPLIT.split(para):
            while len(sent) > size:  # pathological sentence: hard cut on a space
                cut = sent.rfind(" ", 0, size)
                cut = cut if cut > size // 2 else size
                pieces.append(sent[:cut])
                sent = sent[cut:].lstrip()
            if buf and len(buf) + len(sent) + 1 > size:
                pieces.append(buf)
                buf = sent
            else:
                buf = f"{buf} {sent}".strip()
        if buf:
            pieces.append(buf)

    windows: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) + 2 > size:
            windows.append(current)
            tail = _tail(current, overlap)
            current = f"{tail}\n\n{piece}" if tail and len(tail) + len(piece) + 2 <= size * 1.5 else piece
        else:
            current = f"{current}\n\n{piece}".strip()
    if current:
        windows.append(current)
    return windows


def chunk_note(note: Note, size: int = 1200, overlap: int = 200, min_chars: int = 40) -> list[Chunk]:
    """Split a note into retrieval passages that respect headings, pages and code blocks."""
    body = _IMMUTABLE_RE.sub("", note.body) if note.type == "source" else note.body
    groups: list[tuple[str, int | None, list[str]]] = []  # (heading path, page, paragraphs)
    stack: list[tuple[int, str]] = []
    page: int | None = None
    para: list[str] = []
    in_code = False
    in_math = False

    def path() -> str:
        return " › ".join(t for lvl, t in stack if lvl > 1)

    def flush() -> None:
        text = "\n".join(para).strip()
        para.clear()
        if not text:
            return
        key = (path(), page)
        if groups and (groups[-1][0], groups[-1][1]) == key:
            groups[-1][2].append(text)
        else:
            groups.append((key[0], key[1], [text]))

    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            para.append(line)
            continue
        if not in_code and stripped.count("$$") % 2 == 1:
            in_math = not in_math  # a display equation opens or closes on this line
            para.append(line)
            continue
        if not in_code and not in_math:
            m = _HEADING_RE.match(line)
            if m:
                flush()
                level, title = len(m.group(1)), m.group(2).strip()
                pm = _PAGE_HEADING_RE.match(title)
                if pm:
                    page = int(pm.group(1))
                    continue
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, title))
                continue
            pm = _PAGE_MARK_RE.match(stripped)
            if pm:
                flush()
                page = int(pm.group(1))
                continue
            if not stripped:
                flush()
                continue
        para.append(line)
    flush()

    chunks: list[Chunk] = []
    for heading, pg, paragraphs in groups:
        for text in _windows(paragraphs, size, overlap):
            plain = re.sub(r"[#>*_`\-|\s]", "", text)
            if len(plain) < min_chars:
                continue
            chunks.append(Chunk(id=f"{note.slug}::{len(chunks)}", slug=note.slug, title=note.title,
                                note_type=note.type, heading=heading, page=pg, text=text))
    return chunks


# =========================================================================== vector stores
class NumpyStore:
    """Vectors in one float32 .npy file aligned with chunks.jsonl. No extra dependency;
    brute-force cosine is fast up to ~10^5 passages, far beyond a personal research vault."""

    name = "numpy"

    def __init__(self, folder: Path):
        self.path = folder / "vectors.npy"
        self.matrix: np.ndarray | None = np.load(self.path) if self.path.exists() else None

    def rebuild(self, ids: list[str], vectors: np.ndarray | None,
                new_ids: list[str] | None = None, new_vectors: np.ndarray | None = None) -> None:
        """Replace the whole matrix (rows aligned with ``ids``)."""
        self.matrix = vectors
        if vectors is None:
            self.path.unlink(missing_ok=True)
        else:
            tmp = self.path.with_name("vectors.tmp.npy")
            np.save(tmp, vectors.astype(np.float32))
            tmp.replace(self.path)

    def scores(self, query: np.ndarray, ids: list[str]) -> np.ndarray | None:
        if self.matrix is None or len(self.matrix) != len(ids):
            return None
        return self.matrix @ query


class ChromaStore:
    """Vectors in a persistent ChromaDB collection (optional: install the `chroma` extra)."""

    name = "chroma"

    def __init__(self, folder: Path):
        try:
            import chromadb  # type: ignore[import-not-found]
        except ImportError as exc:
            raise LLMError("ChromaDB is not installed: uv tool install --editable . --with chromadb "
                           "(or set [rag] backend = \"numpy\")") from exc
        self.client = chromadb.PersistentClient(path=str(folder / "chroma"))
        self.col = self.client.get_or_create_collection("rbrain", metadata={"hnsw:space": "cosine"})

    def rebuild(self, ids: list[str], vectors: np.ndarray | None,
                new_ids: list[str] | None = None, new_vectors: np.ndarray | None = None) -> None:
        """Delete vectors of chunks that no longer exist and upsert only the new ones."""
        existing = set(self.col.get(include=[])["ids"])
        stale = sorted(existing - set(ids))
        if stale:
            self.col.delete(ids=stale)
        if new_ids and new_vectors is not None and len(new_vectors):
            for i in range(0, len(new_ids), 512):
                self.col.upsert(ids=new_ids[i:i + 512], embeddings=new_vectors[i:i + 512].tolist())

    def scores(self, query: np.ndarray, ids: list[str]) -> np.ndarray | None:
        n = self.col.count()
        if n == 0:
            return None
        res = self.col.query(query_embeddings=[query.tolist()], n_results=min(n, 200), include=["distances"])
        position = {cid: i for i, cid in enumerate(ids)}
        out = np.full(len(ids), -1.0)
        for cid, dist in zip(res["ids"][0], res["distances"][0]):
            if cid in position:
                out[position[cid]] = 1.0 - dist
        return out


# =========================================================================== index
@dataclass
class Hit:
    chunk: Chunk
    score: float
    rank: int = 0


@dataclass
class UpdateStats:
    added: int = 0
    updated: int = 0
    removed: int = 0
    chunks: int = 0
    embedded: int = 0
    notes: int = 0
    embedding_error: str = ""

    def __str__(self) -> str:
        msg = (f"{self.notes} notes → {self.chunks} passages "
               f"(+{self.added} new, ~{self.updated} changed, -{self.removed} removed, "
               f"{self.embedded} passages embedded)")
        return msg + (f"; embeddings unavailable: {self.embedding_error}" if self.embedding_error else "")


def _prefixes(model: str, query_prefix: str, document_prefix: str) -> tuple[str, str]:
    """Some embedding models expect task prefixes; fill them in when not configured."""
    m = model.lower()
    if query_prefix or document_prefix:
        return query_prefix, document_prefix
    if "nomic-embed" in m:
        return "search_query: ", "search_document: "
    if re.search(r"(^|[/:-])e5", m):
        return "query: ", "passage: "
    return "", ""


class RagIndex:
    def __init__(self, vault: Vault, embedder: LLMClient | None = None, model_name: str = "",
                 backend: str = "numpy", chunk_chars: int = 1200, overlap_chars: int = 200,
                 query_prefix: str = "", document_prefix: str = ""):
        self.vault = vault
        self.embedder = embedder
        self.model_name = model_name if embedder else ""
        self.chunk_chars, self.overlap_chars = chunk_chars, overlap_chars
        self.query_prefix, self.document_prefix = _prefixes(model_name, query_prefix, document_prefix)
        self.folder = vault.root / ".rbrain" / "rag"
        self.folder.mkdir(parents=True, exist_ok=True)
        self.backend = backend
        self.store = ChromaStore(self.folder) if backend == "chroma" else NumpyStore(self.folder)
        self.manifest = self._load_manifest()
        self.chunks: list[Chunk] = self._load_chunks()
        self._bm25: BM25 | None = None
        self.embedding_error = ""

    # ------------------------------------------------------------------ persistence
    @property
    def settings(self) -> dict:
        return {"version": INDEX_VERSION, "model": self.model_name, "backend": self.backend,
                "chunk_chars": self.chunk_chars, "overlap_chars": self.overlap_chars,
                "document_prefix": self.document_prefix}

    def _load_manifest(self) -> dict:
        path = self.folder / "manifest.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("settings") == self.settings:
                return data
        return {"settings": self.settings, "notes": {}}

    def _load_chunks(self) -> list[Chunk]:
        path = self.folder / "chunks.jsonl"
        if not path.exists() or not self.manifest["notes"]:
            return []
        return [Chunk(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line]

    def _save(self, vectors: np.ndarray | None, new_chunks: list[Chunk] | None = None,
              new_vectors: np.ndarray | None = None) -> None:
        tmp = self.folder / "chunks.tmp.jsonl"
        tmp.write_text("".join(json.dumps(asdict(c), ensure_ascii=False) + "\n" for c in self.chunks),
                       encoding="utf-8")
        tmp.replace(self.folder / "chunks.jsonl")
        self.store.rebuild([c.id for c in self.chunks], vectors,
                           [c.id for c in new_chunks or []], new_vectors)
        (self.folder / "manifest.json").write_text(json.dumps(self.manifest, indent=1), encoding="utf-8")

    # ------------------------------------------------------------------ update
    def _embed(self, texts: list[str], echo: Callable[[str], None]) -> np.ndarray:
        out: list[list[float]] = []
        for i in range(0, len(texts), 32):
            out.extend(self.embedder.embed(texts[i:i + 32]))  # type: ignore[union-attr]
            if len(texts) > 64:
                echo(f"    embedded {min(i + 32, len(texts))}/{len(texts)} passages")
        mat = np.array(out, dtype=np.float32)
        mat /= np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
        return mat

    def update(self, echo: Callable[[str], None] = lambda *_: None, rebuild: bool = False) -> UpdateStats:
        """Re-chunk and re-embed only the notes that changed since the last update."""
        stats = UpdateStats()
        if rebuild:
            self.manifest = {"settings": self.settings, "notes": {}}
            self.chunks = []
        notes = {n.slug: n for n in self.vault.iter_notes()}
        hashes = {s: hashlib.sha1(n.text().encode("utf-8")).hexdigest() for s, n in notes.items()}
        known: dict = self.manifest["notes"]
        changed = [s for s in notes if known.get(s, {}).get("hash") != hashes[s]]
        removed = [s for s in known if s not in notes]
        stats.notes = len(notes)
        stats.added = sum(1 for s in changed if s not in known)
        stats.updated, stats.removed = len(changed) - stats.added, len(removed)

        old_vectors = None
        if isinstance(self.store, NumpyStore) and self.store.matrix is not None \
                and len(self.store.matrix) == len(self.chunks):
            old_vectors = self.store.matrix
        if not changed and not removed and (self.embedder is None or old_vectors is not None
                                            or self.store.name == "chroma" or not self.chunks):
            stats.chunks = len(self.chunks)
            return stats

        drop = set(changed) | set(removed)
        keep_idx = [i for i, c in enumerate(self.chunks) if c.slug not in drop]
        kept = [self.chunks[i] for i in keep_idx]
        new_chunks: list[Chunk] = []
        for s in changed:
            new_chunks.extend(chunk_note(notes[s], self.chunk_chars, self.overlap_chars))

        vectors = new_vecs = None
        if self.embedder is not None:
            try:
                if old_vectors is None and isinstance(self.store, NumpyStore):
                    kept_vecs = self._embed([c.embed_text(self.document_prefix) for c in kept], echo) \
                        if kept else np.zeros((0, 0), np.float32)
                    stats.embedded += len(kept)
                else:
                    kept_vecs = old_vectors[keep_idx] if old_vectors is not None else None
                new_vecs = self._embed([c.embed_text(self.document_prefix) for c in new_chunks], echo) \
                    if new_chunks else None
                stats.embedded += len(new_chunks)
                parts = [v for v in (kept_vecs, new_vecs) if v is not None and v.size]
                vectors = np.vstack(parts) if parts else None
            except LLMError as exc:
                self.embedding_error = stats.embedding_error = str(exc)
                vectors = new_vecs = None

        self.chunks = kept + new_chunks
        for s in removed:
            known.pop(s, None)
        for s in changed:
            known[s] = {"hash": hashes[s], "chunks": sum(1 for c in new_chunks if c.slug == s)}
        if vectors is None and self.embedder is not None:
            # keep the text index usable; vectors will be retried on the next update
            for s in changed:
                known[s]["hash"] = "pending-embedding"
        self._save(vectors, new_chunks, new_vecs)
        self._bm25 = None
        stats.chunks = len(self.chunks)
        return stats

    # ------------------------------------------------------------------ retrieval
    def _bm25_index(self) -> BM25:
        if self._bm25 is None:
            self._bm25 = BM25([tokenize(f"{c.header}\n{c.text}") for c in self.chunks])
        return self._bm25

    def _dense(self, query: str) -> np.ndarray | None:
        if self.embedder is None or not self.chunks:
            return None
        try:
            q = np.array(self.embedder.embed([self.query_prefix + query])[0], dtype=np.float32)
        except LLMError as exc:
            self.embedding_error = str(exc)
            return None
        q /= np.linalg.norm(q) + 1e-12
        return self.store.scores(q, [c.id for c in self.chunks])

    def search(self, query: str, k: int = 8, scope: str = "all", types: set[str] | None = None,
               queries: list[str] | None = None, max_per_note: int = 3, diversity: float = 0.3,
               pool: int = 50) -> list[Hit]:
        """Hybrid BM25 + dense retrieval with RRF, scope filter, diversification and a per-note cap."""
        if not self.chunks:
            return []
        allowed_types = types or SCOPES.get(scope)
        allowed = np.array([allowed_types is None or c.note_type in allowed_types for c in self.chunks])
        fused = np.zeros(len(self.chunks))
        for q in [query, *(queries or [])]:
            rankings = [self._bm25_index().scores(tokenize(q))]
            dense = self._dense(q)
            if dense is not None:
                rankings.append(dense)
            for scores in rankings:
                order = [i for i in np.argsort(-scores) if scores[i] > 0 and allowed[i]][:pool]
                for rank, i in enumerate(order):
                    fused[i] += 1.0 / (60 + rank)  # Reciprocal Rank Fusion
        candidates = [i for i in np.argsort(-fused) if fused[i] > 0][: max(4 * k, 20)]
        if not candidates:
            return []
        top = fused[candidates[0]]
        tokens = {i: set(tokenize(self.chunks[i].text)) for i in candidates}
        selected: list[int] = []
        per_note: dict[str, int] = {}
        while candidates and len(selected) < k:
            best, best_val = None, -1e9
            for i in candidates:
                if per_note.get(self.chunks[i].slug, 0) >= max_per_note:
                    continue
                redundancy = max((len(tokens[i] & tokens[j]) / (len(tokens[i] | tokens[j]) or 1)
                                  for j in selected), default=0.0)
                val = (1 - diversity) * fused[i] / top - diversity * redundancy
                if val > best_val:
                    best, best_val = i, val
            if best is None:
                break
            selected.append(best)
            candidates.remove(best)
            per_note[self.chunks[best].slug] = per_note.get(self.chunks[best].slug, 0) + 1
        return [Hit(self.chunks[i], float(fused[i]), r + 1) for r, i in enumerate(selected)]

    def note_vectors(self) -> dict[str, np.ndarray]:
        """Mean passage vector per note (used by note-level search, so nothing is embedded twice)."""
        if not isinstance(self.store, NumpyStore) or self.store.matrix is None \
                or len(self.store.matrix) != len(self.chunks):
            return {}
        groups: dict[str, list[int]] = {}
        for i, c in enumerate(self.chunks):
            groups.setdefault(c.slug, []).append(i)
        out = {}
        for slug, idx in groups.items():
            v = self.store.matrix[idx].mean(axis=0)
            out[slug] = v / (np.linalg.norm(v) + 1e-12)
        return out

    def stats(self) -> dict:
        by_type: dict[str, int] = {}
        for c in self.chunks:
            by_type[c.note_type] = by_type.get(c.note_type, 0) + 1
        vec = self.store.matrix if isinstance(self.store, NumpyStore) else None
        return {"notes": len(self.manifest["notes"]), "passages": len(self.chunks), "by_type": by_type,
                "backend": self.backend, "embedding_model": self.model_name or "(none: keyword only)",
                "dimension": int(vec.shape[1]) if vec is not None and vec.ndim == 2 and vec.size else None,
                "chunk_chars": self.chunk_chars, "overlap_chars": self.overlap_chars}


# =========================================================================== context + citations
@dataclass
class Context:
    text: str
    hits: list[Hit]
    dropped: int = 0
    clipped: int = 0
    chars: int = 0

    @property
    def report(self) -> str:
        notes = len({h.chunk.slug for h in self.hits})
        extra = []
        if self.clipped:
            extra.append(f"{self.clipped} clipped")
        if self.dropped:
            extra.append(f"{self.dropped} left out by the budget")
        return (f"{len(self.hits)} passages from {notes} notes, {self.chars:,} characters"
                + (f" ({', '.join(extra)})" if extra else ""))


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit]
    stop = max(cut.rfind(". "), cut.rfind("\n"))
    return (cut[:stop + 1] if stop > limit // 2 else cut).rstrip() + " […]"


def build_context(hits: list[Hit], max_chars: int = 12000, max_passage_chars: int = 2400) -> Context:
    """Number passages [S1].. and fit them into a character budget. Oversized passages are
    clipped rather than dropped (dropping them silently is a classic cause of 'no information'
    answers while the evidence was retrieved)."""
    blocks, used, clipped = [], [], 0
    total = 0
    for hit in hits:
        c = hit.chunk
        n = len(used) + 1
        header = f"[S{n}] {c.location()} — {c.note_type} note [[{c.slug}]]"
        body = _clip(c.text, max_passage_chars)
        clipped += body != c.text
        block = f"{header}\n{body}"
        if total + len(block) > max_chars:
            room = max_chars - total - len(header) - 1
            if room < 300:
                break
            block = f"{header}\n{_clip(body, room)}"
            clipped += 1
        blocks.append(block)
        used.append(hit)
        total += len(block) + 2
    return Context("\n\n".join(blocks), used, dropped=len(hits) - len(used), clipped=clipped, chars=total)


_CITE_RE = re.compile(r"\[(S\d+(?:\s*[,;]\s*S?\d+)*)\]")


@dataclass
class Citations:
    text: str
    cited: list[int]
    invalid: list[int]
    groundedness: float

    def sources_section(self, hits: list[Hit]) -> str:
        if not self.cited:
            return ""
        lines = ["## Sources", ""]
        for n in self.cited:
            c = hits[n - 1].chunk
            lines.append(f"- **S{n}** {c.link(c.location())}")
        return "\n".join(lines)


def link_citations(answer: str, hits: list[Hit]) -> Citations:
    """Turn [S1] / [S1, S3] markers into Obsidian links and measure how grounded the answer is."""
    cited: list[int] = []
    invalid: list[int] = []

    def repl(m: re.Match) -> str:
        parts = []
        for num in map(int, re.findall(r"\d+", m.group(1))):
            if 1 <= num <= len(hits):
                parts.append(hits[num - 1].chunk.link(f"S{num}"))
                if num not in cited:
                    cited.append(num)
            else:
                invalid.append(num)
                parts.append(f"S{num}⚠")
        return "(" + ", ".join(parts) + ")"

    linked = _CITE_RE.sub(repl, answer)
    sentences = [s for s in re.split(r"(?<=[.!?])\s+|\n+", answer)
                 if len(s.split()) >= 6 and not s.lstrip().startswith(("#", "**Gaps", "Gaps"))]
    def valid(sentence: str) -> bool:  # grounded = carries at least one *valid* citation
        return any(1 <= int(n) <= len(hits) for m in _CITE_RE.finditer(sentence)
                   for n in re.findall(r"\d+", m.group(1)))
    grounded = sum(1 for s in sentences if valid(s)) / len(sentences) if sentences else 1.0
    return Citations(linked, sorted(cited), sorted(set(invalid)), round(grounded, 2))


# =========================================================================== LLM helpers
EXPAND_SYSTEM = (
    "You rewrite a research question into alternative search queries for a bilingual "
    "(English/Spanish) research vault. Return ONLY JSON: {\"queries\": [\"...\", \"...\", \"...\"]}. "
    "Give 3 short queries: one with technical synonyms, one more specific, and one translated into "
    "the other language (English <-> Spanish). Keep mathematical terms precise."
)
RERANK_SYSTEM = (
    "You judge how useful each numbered passage is for answering a research question. "
    "Score 0 = irrelevant, 1 = related topic, 2 = partially answers, 3 = directly answers. "
    "Return ONLY JSON: {\"scores\": [s1, s2, ...]} with one integer per passage, in order."
)
CONDENSE_SYSTEM = (
    "Rewrite the user's follow-up message as a single standalone question that can be understood "
    "without the conversation. Keep the user's language and technical terms. Return only the question."
)
