"""The research workflows: ingest, segment, synthesize, explain, deduce, ask/chat, lint.

Each workflow follows the same shape:

    gather context from the vault (search / RAG)  ->  run a skill on an LLM (JSON)
    ->  validate / verify (code, SymPy)            ->  render notes  ->  update index, log, RAG

so that the knowledge base *compounds*: every new note is linked to what already exists and
becomes retrievable passage by passage.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import itertools
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import render
from .config import Config
from .extract import chunk_text, guess_title, head_and_tail, read_source
from .llm import LLMClient, LLMError, chat_json, make_client, strip_think
from .logic import ArgumentCheck, FormulaError, check_argument
from .rag import (CONDENSE_SYSTEM, EXPAND_SYSTEM, RERANK_SYSTEM, Citations, Context, Hit, RagIndex,
                  UpdateStats, build_context, link_citations)
from .search import SearchIndex, tokenize
from .skills import load_skill
from .vault import Note, Vault, link, slugify, today, wikilinks

EXCERPT = 1500  # characters of each note given to the model as context


def excerpt(note: Note, limit: int = EXCERPT) -> str:
    body = re.sub(r"\n{3,}", "\n\n", note.body.strip())
    return body[:limit] + ("…" if len(body) > limit else "")


def context_block(notes: list[Note], limit: int = EXCERPT) -> str:
    return "\n\n".join(f"### [[{n.slug}]] {n.title}  (type: {n.type or 'note'})\n{excerpt(n, limit)}"
                       for n in notes)


def first_statement(note: Note) -> str:
    """The main statement of a note: its summary field, else its first paragraph."""
    if note.meta.get("summary"):
        return str(note.meta["summary"])
    for para in re.split(r"\n\s*\n", note.body):
        para = para.strip()
        if para and not para.startswith(("#", ">", "|", "-", "**")):
            return para
    return note.title


@dataclass
class Answer:
    """A grounded answer: text with linked citations, the context that produced it, diagnostics."""

    question: str
    text: str
    raw: str
    context: Context
    citations: Citations
    note: Note | None = None

    @property
    def diagnostics(self) -> str:
        parts = [f"context: {self.context.report}", f"cited passages: {len(self.citations.cited)}",
                 f"groundedness: {self.citations.groundedness:.0%}"]
        if self.citations.invalid:
            parts.append("invalid citations: " + ", ".join(f"S{n}" for n in self.citations.invalid))
        return " · ".join(parts)


@dataclass
class LintReport:
    path: Path
    broken: dict[str, list[str]] = field(default_factory=dict)
    orphans: list[str] = field(default_factory=list)
    unsourced: list[str] = field(default_factory=list)
    duplicates: list[tuple[str, str]] = field(default_factory=list)
    invalid_arguments: list[str] = field(default_factory=list)
    stale_index: bool = False


class Brain:
    def __init__(self, config: Config, client_factory: Callable[..., LLMClient] = make_client,
                 echo: Callable[[str], None] = print):
        self.cfg = config
        self.vault = Vault(config.vault)
        self._factory = client_factory
        self._clients: dict[tuple, LLMClient] = {}
        self._embedder: LLMClient | None = None
        self._rag: RagIndex | None = None
        self.echo = echo

    # ------------------------------------------------------------------ plumbing
    def client(self, skill: str) -> LLMClient:
        p = self.cfg.profile_for(skill)
        key = (p.provider, p.model, p.base_url)
        if key not in self._clients:
            self._clients[key] = self._factory(p)
        return self._clients[key]

    def embedder(self) -> LLMClient | None:
        if not self.cfg.embeddings.enabled:
            return None
        if self._embedder is None:
            self._embedder = self._factory(self.cfg.embeddings.as_profile())
        return self._embedder

    def rag(self) -> RagIndex:
        """The passage-level RAG index (vectors in .rbrain/rag/)."""
        if self._rag is None:
            e, r = self.cfg.embeddings, self.cfg.rag
            self._rag = RagIndex(self.vault, self.embedder(), e.model if e.enabled else "",
                                 backend=r.backend, chunk_chars=r.chunk_chars, overlap_chars=r.overlap_chars,
                                 query_prefix=e.query_prefix, document_prefix=e.document_prefix)
        return self._rag

    def refresh_rag(self, force: bool = False, rebuild: bool = False) -> UpdateStats | None:
        """Incrementally re-index changed notes (automatic unless [rag] auto_update = false)."""
        if not (self.cfg.rag.auto_update or force or rebuild):
            return None
        stats = self.rag().update(echo=self.echo, rebuild=rebuild)
        if stats.embedding_error:
            self.echo(f"  ! embeddings unavailable, keyword retrieval only: {stats.embedding_error}")
        return stats

    def search_index(self, embeddings: bool = True) -> SearchIndex:
        """Note-level search; reuses the RAG passage vectors so nothing is embedded twice."""
        embedder = self.embedder() if embeddings else None
        vectors: dict = {}
        prefix = ""
        if embedder is not None:
            self.refresh_rag()
            vectors = self.rag().note_vectors()
            prefix = self.rag().query_prefix
        return SearchIndex(self.vault, embedder, self.cfg.embeddings.model, note_vectors=vectors,
                           query_prefix=prefix)

    # ------------------------------------------------------------------ retrieval (RAG)
    def retrieve(self, query: str, k: int | None = None, scope: str | None = None,
                 types: set[str] | None = None, expand: bool | None = None,
                 rerank: bool | None = None) -> list[Hit]:
        """Hybrid passage retrieval, optionally with query expansion and LLM reranking."""
        r = self.cfg.rag
        k = k or r.top_k
        expand = r.expand_queries if expand is None else expand
        rerank = r.rerank if rerank is None else rerank
        self.refresh_rag()
        queries = self.expand_query(query) if expand else []
        hits = self.rag().search(query, k=2 * k if rerank else k, scope=scope or r.scope, types=types,
                                 queries=queries, max_per_note=r.max_per_note, diversity=r.diversity)
        if rerank and len(hits) > 1:
            hits = self.rerank(query, hits)
        return hits[:k]

    def expand_query(self, query: str) -> list[str]:
        """Multi-query: technical synonyms, a more specific variant and an EN<->ES translation
        (keyword search cannot match a Spanish question to English notes on its own)."""
        try:
            d = chat_json(self.client("query-brain"), EXPAND_SYSTEM, f"Question: {query}", tag="rag-expand")
        except LLMError as exc:
            self.echo(f"  ! query expansion skipped: {exc}")
            return []
        variants = [str(q).strip() for q in d.get("queries", []) if str(q).strip() and str(q).strip() != query]
        if variants:
            self.echo("  · query variants: " + " | ".join(variants[:4]))
        return variants[:4]

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        """Listwise LLM reranking; falls back to the retrieval order on any problem."""
        listing = "\n\n".join(f"[{i}] {h.chunk.location()}\n{h.chunk.text[:600]}" for i, h in enumerate(hits, 1))
        try:
            d = chat_json(self.client("query-brain"), RERANK_SYSTEM,
                          f"Question: {query}\n\nPassages:\n{listing}", tag="rag-rerank")
            scores = [float(x) for x in d.get("scores", [])]
        except (LLMError, TypeError, ValueError) as exc:
            self.echo(f"  ! reranking skipped: {exc}")
            return hits
        if len(scores) != len(hits):
            return hits
        order = sorted(range(len(hits)), key=lambda i: (-scores[i], i))
        kept = [hits[i] for i in order if scores[i] > 0]
        return kept or hits

    def condense(self, message: str, history: list[tuple[str, str]]) -> str:
        """Turn a chat follow-up ("and for networks?") into a standalone question for retrieval."""
        if not history:
            return message
        convo = "\n".join(f"User: {q}\nAssistant: {a[:400]}" for q, a in history[-3:])
        try:
            out = strip_think(self.client("query-brain").chat(
                CONDENSE_SYSTEM, f"Conversation:\n{convo}\n\nFollow-up: {message}", tag="rag-condense"))
        except LLMError:
            return message
        out = out.strip().splitlines()[0].strip() if out.strip() else ""
        return out or message

    def run_skill(self, name: str, user_prompt: str, *, stage: str | None = None,
                  extra: str = "") -> dict:
        skill = load_skill(name, self.vault.root)
        schema = skill.extra_schema(stage) if stage else None
        system = skill.system_prompt(self.cfg.language, extra, schema=schema)
        tag = f"{name}.{stage}" if stage else name
        p = self.cfg.profile_for(name)
        self.echo(f"  · {tag} → {p.provider}:{p.model}")
        return chat_json(self.client(name), system, user_prompt, tag=tag)

    def finish(self, operation: str, title: str, notes: list[Note]) -> None:
        self.vault.update_index()
        self.vault.append_log(operation, title, notes)
        if self.cfg.rag.auto_update:
            self.refresh_rag()

    # ------------------------------------------------------------------ ingest
    def ingest(self, source: str | Path, title: str | None = None, segment: bool = True,
               max_chunks: int = 6, use_marker: bool = False) -> list[Note]:
        """Karpathy's *ingest*: keep an immutable copy, write a summary page, segment into atoms."""
        src = Path(source).expanduser().resolve()
        text, kind = read_source(src, use_marker=use_marker)
        if not text.strip():
            raise ValueError(f"No text could be extracted from {src} (scanned PDF? try --marker).")
        sha = hashlib.sha256(src.read_bytes()).hexdigest()
        for existing in self.vault.iter_notes({"source"}):
            if existing.meta.get("sha256") == sha:
                self.echo(f"Already ingested as {existing.path.relative_to(self.vault.root)}; skipping.")
                return [existing]
        title = title or guess_title(text, src.stem)

        meta = {"original": str(src), "sha256": sha, "format": kind, "ingested": dt.date.today()}
        if kind not in {"md", "markdown", "txt"}:
            assets = self.vault.dir_for("source") / "assets"
            assets.mkdir(parents=True, exist_ok=True)
            copy = assets / f"{slugify(title)}{src.suffix.lower()}"
            shutil.copy2(src, copy)
            meta["attachment"] = f"[[{copy.name}]]"
        body = (f"# {title}\n\n> [!warning] Immutable source\n> Extracted text of the original. "
                f"Do not edit: the summary, atoms and concepts link back here.\n\n{text}")
        source_note = self.vault.write("source", title, meta, body)
        created = [source_note]

        prompt = (f"Source title: {title}\nSource note: [[{source_note.slug}]]\n\n<source>\n"
                  f"{head_and_tail(text, self.cfg.chunk_chars * 2)}\n</source>")
        d = self.run_skill("ingest-source", prompt)
        summary_title = f"Summary — {title}"
        summary = self.vault.write(
            "summary", summary_title,
            {"source": source_note.link(), "summary": d.get("summary", ""),
             "tags": [slugify(t) for t in d.get("tags", [])][:8]},
            render.summary_body({**d, "title": summary_title}, source_note.link()),
        )
        created.append(summary)

        if segment:
            atoms = self.segment(text, source=source_note, max_chunks=max_chunks, log=False)
            created += atoms
            atom_list = "\n".join(f"- {a.link()}" for a in atoms) or "_No atoms extracted._"
            summary = self.vault.write("summary", summary.title, summary.meta,
                                       summary.body.replace("_Filled in by idea segmentation._", atom_list),
                                       path=summary.path)
        self.finish("ingest", title, created)
        return created

    # ------------------------------------------------------------------ segment
    def segment(self, text: str, source: Note | None = None, max_chunks: int = 6,
                log: bool = True, label: str | None = None) -> list[Note]:
        """Idea segmentation: split a text into atomic, linkable idea notes."""
        index = self.search_index()
        chunks = chunk_text(text, self.cfg.chunk_chars)
        if len(chunks) > max_chunks:
            self.echo(f"  ! {len(chunks)} chunks; processing the first {max_chunks} (use --max-chunks).")
            chunks = chunks[:max_chunks]
        source_link = source.link() if source else None
        written: list[Note] = []
        reserved: set[str] = set()
        for i, chunk in enumerate(chunks, start=1):
            prompt = (
                f"Source: {source.title if source else (label or 'personal note')}\n"
                f"Chunk {i} of {len(chunks)}.\n"
                f"Extract at most {self.cfg.max_atoms_per_chunk} atomic ideas.\n\n<text>\n{chunk}\n</text>"
            )
            d = self.run_skill("segment-idea", prompt)
            atoms = [a for a in d.get("atoms", []) if isinstance(a, dict) and a.get("statement")]
            atoms = atoms[: self.cfg.max_atoms_per_chunk]
            paths = [self.vault.unique_path("atom", a.get("title") or a["statement"][:60], reserved=reserved)
                     for a in atoms]
            titles = [a.get("title") or a["statement"][:60] for a in atoms]
            for j, (a, path) in enumerate(zip(atoms, paths)):
                hits = index.search(a["statement"], k=3, types={"atom", "concept", "idea", "summary"},
                                    exclude={source.slug} if source else None)
                related = [h.link() for h, _ in hits]
                deps = []
                for k in a.get("depends_on", []) or []:
                    if isinstance(k, int) and 0 <= k < len(atoms) and k != j:
                        deps.append(link(paths[k].stem, titles[k]))
                meta = {
                    "kind": a.get("kind", "claim"),
                    "confidence": a.get("confidence", "medium"),
                    "source": source_link,
                    "summary": a["statement"],
                    "tags": [slugify(c) for c in a.get("concepts", [])][:5],
                }
                written.append(self.vault.write("atom", titles[j], meta,
                                                render.atom_body(a, source_link, related, deps), path=path))
        if log:
            self.finish("segment", source.title if source else (label or "text"), written)
        return written

    # ------------------------------------------------------------------ synthesize
    def synthesize(self, refs: list[str], query: str | None = None, n: int = 3) -> list[Note]:
        """Construct new ideas by applying creative operators to existing notes."""
        parents: list[Note] = []
        for r in refs:
            note = self.vault.resolve(r)
            if note is None:
                raise FileNotFoundError(f"No note matches '{r}'")
            parents.append(note)
        index = self.search_index()
        if query:
            exclude = {p.slug for p in parents}
            for note, _ in index.search(query, k=max(2, 6 - len(parents)),
                                        types={"atom", "concept", "idea", "summary", "argument"},
                                        exclude=exclude):
                parents.append(note)
        if len(parents) < 2:
            raise ValueError("Synthesis needs at least two notes: pass note names and/or --query.")
        parent_slugs = {p.slug for p in parents}
        focus = query or " ".join(p.title for p in parents)
        existing = [h for h, _ in index.search(focus, k=5, types={"idea"}) if h.slug not in parent_slugs]
        prompt = (
            f"Research focus: {query or 'find the most promising combinations of these notes'}\n"
            f"Generate exactly {n} candidate ideas. In `parents`, use only slugs listed below.\n\n"
            f"## Parent notes\n{context_block(parents)}\n\n"
            "## Ideas already in the vault (do not repeat them; build on them if useful)\n"
            + ("\n".join(f"- [[{e.slug}]] {e.title}: {first_statement(e)[:200]}" for e in existing)
               or "- none")
        )
        d = self.run_skill("synthesize-ideas", prompt)
        written = []
        for idea in (d.get("ideas") or [])[:n]:
            if not isinstance(idea, dict) or not idea.get("statement"):
                continue
            chosen = [s for s in idea.get("parents", []) if s in parent_slugs] or sorted(parent_slugs)
            parent_links = [p.link() for p in parents if p.slug in chosen]
            similar = [h.link() for h, _ in index.search(idea["statement"], k=3, types={"idea"})]
            scores = idea.get("scores", {}) or {}
            meta = {
                "status": "seed",
                "operator": idea.get("operator", "combination"),
                "novelty": scores.get("novelty"),
                "feasibility": scores.get("feasibility"),
                "impact": scores.get("impact"),
                "parents": parent_links,
                "summary": idea["statement"],
                "tags": ["idea", slugify(str(idea.get("operator", "combination")))],
            }
            written.append(self.vault.write("idea", idea.get("title") or idea["statement"][:60], meta,
                                            render.idea_body(idea, parent_links, similar)))
        self.finish("synthesize", query or ", ".join(refs), written)
        return written

    # ------------------------------------------------------------------ explain
    def explain(self, concept: str, level: str = "graduate") -> Note:
        """Write (or regenerate) a layered concept page: intuition → formal → example → test."""
        concept = concept.strip()
        own = slugify(concept)
        hits = [h for h in self.retrieve(concept, k=8) if h.chunk.slug != own]
        ctx = build_context(hits, min(self.cfg.rag.max_context_chars, 8000), max_passage_chars=900)
        prompt = (
            f"Concept to explain: {concept}\nAudience level: {level}\n\n"
            "Passages from the vault that may be relevant (ground the explanation in them when "
            "possible; list in `vault_links` the note slugs shown after [[...]] only if they "
            "genuinely relate):\n" + (ctx.text or "(none)")
        )
        d = self.run_skill("explain-concept", prompt)
        known = {h.chunk.slug for h in ctx.hits}
        vault_links = [link(s) for s in dict.fromkeys(d.get("vault_links", [])) if s in known]
        existing = self.vault.resolve(concept)
        path = existing.path if existing is not None and existing.type == "concept" else None
        aliases = [a for a in {d.get("title", ""), concept} if a and slugify(a) != slugify(concept)]
        meta = {"level": level, "summary": d.get("one_line", ""), "aliases": aliases,
                "tags": ["concept", *[slugify(t) for t in d.get("tags", [])][:5]]}
        body = render.concept_body({**d, "title": concept}, vault_links)
        note = self.vault.write("concept", concept, meta, body, path=path)
        self.finish("explain", concept, [note])
        return note

    # ------------------------------------------------------------------ deduce
    def deduce(self, premises: list[str], goal: str, rounds: int = 2) -> tuple[Note, ArgumentCheck]:
        """Deductive reasoning with a symbolic check and a generate→verify→revise loop."""
        stated = []
        for p in premises:
            note = self.vault.resolve(p) if (p.startswith("[[") or p.endswith(".md")) else None
            if note is not None:
                stated.append({"text": first_statement(note), "source": note.slug})
            else:
                stated.append({"text": p, "source": None})
        hits = self.retrieve(goal + " " + " ".join(s["text"] for s in stated), k=6)
        evidence = build_context(hits, 6000, max_passage_chars=700)
        prem_md = "\n".join(f"P{i}. {s['text']}" + (f"  (from [[{s['source']}]])" if s["source"] else "")
                            for i, s in enumerate(stated, 1))
        prompt = (
            f"## Premises\n{prem_md}\n\n## Goal (conclusion to establish)\n{goal}\n\n"
            "## Related passages in the vault (use them to judge soundness, not as extra premises; "
            "refer to them as [[note-slug]])\n" + (evidence.text or "(none)")
        )
        d = self.run_skill("deductive-reasoning", prompt)
        for i, s in enumerate(stated):  # keep note provenance even if the model drops it
            if i < len(d.get("premises", [])) and s["source"]:
                d["premises"][i]["source"] = s["source"]

        def formulas() -> tuple[list[str], str]:
            return ([str(p.get("formula", "")) for p in d.get("premises", [])],
                    str((d.get("conclusion") or {}).get("formula", "")))

        history: list[str] = []
        added: list[dict] = []
        try:
            fs, goal_f = formulas()
            check = check_argument(fs, goal_f)
        except FormulaError as exc:
            history.append(f"Round 0: formula syntax error ({exc}); asking the model to fix it.")
            fix = self.run_skill("deductive-reasoning", prompt + f"\n\nYour formulas did not parse: {exc}. "
                                 "Use only atoms, ~, &, |, ->, <-> and parentheses.")
            d.update(fix)
            fs, goal_f = formulas()
            check = check_argument(fs, goal_f)
        history.append(f"Round 0: {check.verdict}")

        r = 0
        while check.premises_consistent and not check.valid and r < rounds:
            r += 1
            ce = ", ".join(f"{k}={'T' if v else 'F'}" for k, v in sorted((check.counterexample or {}).items()))
            revise_prompt = (
                f"The argument below is INVALID. Counter-model (all premises true, conclusion false): {ce}\n\n"
                f"Dictionary: {d.get('dictionary', {})}\n"
                f"Premises: {fs + [a['formula'] for a in added]}\nConclusion: {goal_f}\n\n"
                "Propose the MINIMAL missing premise(s) that would make it valid, state each in plain "
                "language, judge how plausible each is in this domain, and give a corrected derivation."
            )
            rv = self.run_skill("deductive-reasoning", revise_prompt, stage="revise")
            new = [m for m in rv.get("missing_premises", []) if isinstance(m, dict) and m.get("formula")]
            d.setdefault("dictionary", {}).update(rv.get("new_atoms", {}) or {})
            if rv.get("derivation"):
                d["derivation"] = rv["derivation"]
            try:
                candidate = check_argument(fs + [a["formula"] for a in added + new], goal_f)
            except FormulaError as exc:
                history.append(f"Round {r}: proposed premise did not parse ({exc}).")
                continue
            added += new
            check = candidate
            history.append(f"Round {r}: added {', '.join('`' + m['formula'] + '`' for m in new) or 'nothing'} "
                           f"→ {check.verdict}")

        verdict = "vacuous" if not check.premises_consistent else ("valid" if check.valid else "invalid")
        title = d.get("title") or f"Argument — {goal[:70]}"
        meta = {
            "verdict": verdict,
            "repaired": bool(added),
            "premise_sources": [link(s["source"]) for s in stated if s["source"]],
            "summary": f"{goal} ({verdict}{', with added premises' if added else ''})",
            "tags": ["argument", verdict],
        }
        note = self.vault.write("argument", title, meta, render.argument_body(d, check, added, history))
        self.finish("deduce", goal, [note])
        return note, check

    # ------------------------------------------------------------------ ask
    def ask(self, question: str, save: bool = False, k: int | None = None, scope: str | None = None,
            expand: bool | None = None, rerank: bool | None = None,
            history: list[tuple[str, str]] | None = None) -> Answer:
        """RAG answer: retrieve passages, answer with [S#] citations, link them, check grounding.
        With ``save`` the answer is filed in 70-answers/, so questions also compound."""
        hits = self.retrieve(question, k=k, scope=scope, expand=expand, rerank=rerank)
        ctx = build_context(hits, self.cfg.rag.max_context_chars)
        system = load_skill("query-brain", self.vault.root).system_prompt(self.cfg.language)
        convo = ""
        if history:
            convo = "## Conversation so far\n" + "\n".join(
                f"Q: {q}\nA: {a[:600]}" for q, a in history[-3:]) + "\n\n"
        prompt = (f"{convo}Question: {question}\n\n## Passages retrieved from the vault\n"
                  + (ctx.text or "(no relevant passages found)"))
        p = self.cfg.profile_for("query-brain")
        self.echo(f"  · query-brain → {p.provider}:{p.model} ({ctx.report})")
        raw = strip_think(self.client("query-brain").chat(system, prompt, tag="query-brain"))
        cites = link_citations(raw, ctx.hits)
        text = cites.text.rstrip()
        if cites.cited:
            text += "\n\n" + cites.sources_section(ctx.hits)
        answer = Answer(question, text, raw, ctx, cites)
        if save:
            self.save_answer(answer)
        else:
            self.vault.append_log("ask", question)
        return answer

    def save_answer(self, answer: Answer) -> Note:
        """File an answer in 70-answers/ (its citations become links in the graph)."""
        first = next((ln for ln in answer.raw.strip().splitlines() if ln.strip()), "")
        body = f"# {answer.question}\n\n{answer.text}\n\n> [!info] Retrieval\n> {answer.diagnostics}\n"
        answer.note = self.vault.write(
            "answer", answer.question[:90],
            {"question": answer.question, "groundedness": answer.citations.groundedness,
             "cited": [answer.context.hits[n - 1].chunk.link() for n in answer.citations.cited],
             "summary": re.sub(r"\[S\d+[^\]]*\]", "", first)[:200]},
            body)
        self.finish("ask", answer.question, [answer.note])
        return answer.note

    # ------------------------------------------------------------------ lint
    def lint(self, use_llm: bool = False) -> LintReport:
        """Health check: broken links (missing concepts), orphans, unsourced claims, duplicates."""
        notes = list(self.vault.iter_notes())
        all_slugs = {p.stem for p in self.vault.root.rglob("*.md")}
        all_slugs |= {p.name for p in self.vault.root.rglob("*") if p.is_file()}  # attachments
        inbound: dict[str, int] = defaultdict(int)
        broken: dict[str, list[str]] = defaultdict(list)
        for n in notes:
            targets = wikilinks(n.body) | wikilinks(" ".join(map(str, n.meta.values())))
            for t in targets:
                if t in all_slugs:
                    inbound[t] += 1
                else:
                    broken[t].append(n.slug)
        report = LintReport(path=self.vault.root / "lint-report.md")
        report.broken = dict(sorted(broken.items(), key=lambda kv: -len(kv[1])))
        report.orphans = [n.slug for n in notes if inbound[n.slug] == 0 and n.type != "source"]
        report.unsourced = [n.slug for n in notes if n.type == "atom" and not n.meta.get("source")
                            and n.meta.get("kind") in {"claim", "result", "evidence"}]
        report.invalid_arguments = [n.slug for n in notes if n.type == "argument"
                                    and n.meta.get("verdict") in {"invalid", "vacuous"}]
        comparable = [n for n in notes if n.type in {"atom", "concept", "idea"}][:1500]
        tok = {n.slug: set(tokenize(n.title)) for n in comparable}
        for a, b in itertools.combinations(comparable, 2):
            ta, tb = tok[a.slug], tok[b.slug]
            if len(ta) >= 3 and len(tb) >= 3 and len(ta & tb) / len(ta | tb) >= 0.8:
                report.duplicates.append((a.slug, b.slug))
        index_path = self.vault.root / "index.md"
        if index_path.exists():
            report.stale_index = any(n.path.stat().st_mtime > index_path.stat().st_mtime for n in notes)

        out = [f"# Lint report ({today()})", "",
               f"{len(notes)} notes checked. Re-run with `rbrain lint` after fixing.", ""]
        out += ["## Missing pages (linked but not written)",
                "_Concepts mentioned in several notes but without their own page. "
                "Fill one with `rbrain explain \"<name>\"`._", ""]
        out += [f"- `{t}` ← {len(src)} note(s): {', '.join(link(s) for s in src[:5])}"
                for t, src in list(report.broken.items())[:40]] or ["_None._"]
        out += ["", "## Orphans (no inbound links)", ""]
        out += [f"- {link(s)}" for s in report.orphans[:60]] or ["_None._"]
        out += ["", "## Unsourced claims", ""]
        out += [f"- {link(s)}" for s in report.unsourced[:60]] or ["_None._"]
        out += ["", "## Possible duplicates (merge or link them)", ""]
        out += [f"- {link(a)} ≈ {link(b)}" for a, b in report.duplicates[:40]] or ["_None._"]
        out += ["", "## Arguments that are not (yet) valid", ""]
        out += [f"- {link(s)}" for s in report.invalid_arguments] or ["_None._"]
        out += ["", f"**Index up to date:** {'no — run `rbrain index`' if report.stale_index else 'yes'}"]

        if use_llm and notes:
            index_text = index_path.read_text(encoding="utf-8")[:6000] if index_path.exists() else ""
            d = self.run_skill("lint-brain", "## Lint findings\n" + "\n".join(out) +
                               "\n\n## Index of the vault\n" + index_text)
            out += ["", "## Suggested research questions", ""]
            for q in d.get("questions", []) or []:
                rel = ", ".join(link(s) for s in q.get("related", []) if s in all_slugs)
                out.append(f"- [ ] **{q.get('question', '')}** — {q.get('why', '')}" + (f" ({rel})" if rel else ""))
            if d.get("suggested_sources"):
                out += ["", "## Sources worth looking for", ""] + [f"- {s}" for s in d["suggested_sources"]]
        report.path.write_text("\n".join(out) + "\n", encoding="utf-8")
        self.vault.append_log("lint", f"{len(report.broken)} missing pages, {len(report.orphans)} orphans")
        return report
