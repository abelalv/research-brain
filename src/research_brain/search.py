"""Hybrid retrieval over the vault: BM25 (always) + embeddings (optional), fused with RRF.

At small scale Karpathy's advice holds: the LLM can read ``index.md`` and open pages.
Past a few hundred notes you want real search, and the deterministic linking step of the
segmentation skill ("which existing notes are related to this new atom?") needs it too.

Embeddings are cached in ``.rbrain/embeddings.json`` keyed by path and modification time,
so only new or edited notes are re-embedded.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

from .llm import LLMClient, LLMError
from .vault import Note, Vault

# Small bilingual stop-word list (English + Spanish), enough for BM25 on research notes.
STOPWORDS = set("""
a an and are as at be by for from has have in is it its of on or that the this to was were
will with which what when where who how why can could should would may might not no but if
then than so such these those there their they them we our you your i he she his her
el la los las un una unos unas y o de del al en con por para que se es son lo como mas pero
sus su le les ya muy sin sobre entre cuando donde este esta estos estas ese esa eso
""".split())


def tokenize(text: str) -> list[str]:
    """Lowercase, fold accents, split on non-alphanumerics, drop stop-words."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii").lower()
    return [t for t in re.findall(r"[a-z0-9]+", text) if len(t) > 1 and t not in STOPWORDS]


class BM25:
    """Okapi BM25 with the usual k1/b parameters."""

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(d) for d in docs]
        self.lengths = np.array([len(d) for d in docs], dtype=float)
        self.avgdl = float(self.lengths.mean()) if len(docs) else 0.0
        df: Counter = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(len(self.docs))
        for i, d in enumerate(self.docs):
            denom_norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / (self.avgdl or 1))
            s = 0.0
            for t in query:
                f = d.get(t, 0)
                if f:
                    s += self.idf.get(t, 0.0) * f * (self.k1 + 1) / (f + denom_norm)
            out[i] = s
        return out


def note_text(note: Note, max_chars: int = 4000) -> str:
    tags = " ".join(map(str, note.meta.get("tags", []) or []))
    return f"{note.title}\n{note.meta.get('summary', '')}\n{tags}\n{note.body}"[:max_chars]


class SearchIndex:
    """Note-level search (used to link new notes to related ones and by `rbrain search`).

    When the RAG index is available, note vectors are the mean of their passage vectors, so
    nothing is embedded twice; otherwise notes are embedded here and cached.
    """

    def __init__(self, vault: Vault, embedder: LLMClient | None = None, model_name: str = "",
                 note_vectors: dict | None = None, query_prefix: str = ""):
        self.vault = vault
        self.embedder = embedder
        self.model_name = model_name
        self.note_vectors = note_vectors or {}
        self.query_prefix = query_prefix
        self.notes: list[Note] = list(vault.iter_notes())
        self.bm25 = BM25([tokenize(note_text(n)) for n in self.notes]) if self.notes else None
        self._vectors: np.ndarray | None = None
        self.embedding_error: str | None = None

    # ---------------------------------------------------------------- embeddings
    @property
    def cache_path(self) -> Path:
        return self.vault.root / ".rbrain" / "embeddings.json"

    def ensure_embeddings(self) -> np.ndarray | None:
        """Embed every note (using the cache). Returns None if embeddings are unavailable."""
        if self.embedder is None or not self.notes:
            return None
        if self._vectors is not None:
            return self._vectors
        if self.note_vectors:  # notes without passages (very short) rely on BM25 only
            dim = len(next(iter(self.note_vectors.values())))
            self._vectors = np.array([self.note_vectors.get(n.slug, np.zeros(dim)) for n in self.notes],
                                     dtype=float)
            return self._vectors
        cache: dict = {}
        if self.cache_path.exists():
            try:
                cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                cache = {}
        keys, todo = [], []
        for n in self.notes:
            key = str(n.path.relative_to(self.vault.root))
            stamp = f"{n.path.stat().st_mtime:.0f}:{self.model_name}"
            keys.append(key)
            if cache.get(key, {}).get("stamp") != stamp:
                todo.append((key, stamp, note_text(n, 2000)))
        try:
            for i in range(0, len(todo), 16):
                batch = todo[i:i + 16]
                vecs = self.embedder.embed([t for _, _, t in batch])
                for (key, stamp, _), vec in zip(batch, vecs):
                    cache[key] = {"stamp": stamp, "vec": vec}
        except LLMError as exc:
            self.embedding_error = str(exc)  # fall back to BM25 only
            return None
        live = {k: cache[k] for k in keys}
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(live), encoding="utf-8")
        mat = np.array([live[k]["vec"] for k in keys], dtype=float)
        mat /= np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
        self._vectors = mat
        return mat

    # ---------------------------------------------------------------- query
    def search(self, query: str, k: int = 8, types: set[str] | None = None,
               exclude: set[str] | None = None) -> list[tuple[Note, float]]:
        """Return up to ``k`` (note, score) pairs, fusing BM25 and cosine rankings with RRF."""
        if not self.notes or self.bm25 is None:
            return []
        exclude = exclude or set()
        mask = [(types is None or n.type in types) and n.slug not in exclude for n in self.notes]
        rankings: list[np.ndarray] = []

        bm = self.bm25.scores(tokenize(query))
        if bm.max() > 0:
            rankings.append(bm)
        vecs = self.ensure_embeddings()
        if vecs is not None:
            q = np.array(self.embedder.embed([self.query_prefix + query])[0], dtype=float)  # type: ignore[union-attr]
            q /= np.linalg.norm(q) + 1e-12
            rankings.append(vecs @ q)
        if not rankings:
            return []

        fused = np.zeros(len(self.notes))
        for scores in rankings:
            order = np.argsort(-scores)
            for rank, idx in enumerate(order):
                if scores[idx] > 0:
                    fused[idx] += 1.0 / (60 + rank)  # Reciprocal Rank Fusion
        results = [(self.notes[i], float(fused[i])) for i in np.argsort(-fused)
                   if mask[i] and fused[i] > 0]
        return results[:k]
