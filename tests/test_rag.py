"""RAG: chunking, incremental index, hybrid retrieval, context budget, citations, chat, CLI."""

import builtins
import re

import pytest
from conftest import EXAMPLES

from research_brain.cli import main
from research_brain.config import LLMProfile, load_config
from research_brain.llm import FakeClient, LLMError
from research_brain.rag import Chunk, Hit, RagIndex, build_context, chunk_note, link_citations
from research_brain.vault import Note, Vault
from research_brain.workflows import Brain

SOURCE = EXAMPLES / "sources" / "pinn-sir-inverse-problem.md"


def note(body: str, slug: str = "n", type_: str = "atom", title: str = "T") -> Note:
    from pathlib import Path
    return Note(Path(f"/x/{slug}.md"), {"title": title, "type": type_}, body)


# ------------------------------------------------------------------ chunking
def test_chunks_follow_headings_and_pages():
    body = ("# Title\n\nintro paragraph that is long enough to be kept as a passage.\n\n"
            "## Methods\n\n### Loss\n\nThe loss combines data misfit and the residual of the ODE system.\n\n"
            "## Page 3\n\nText on page three about identifiability of the reporting rate.\n\n"
            "[page 4]\n\nLegacy page marker text about collocation points in PINNs.")
    chunks = chunk_note(note(body, type_="source"), size=200, overlap=0)
    assert [c.heading for c in chunks] == ["", "Methods › Loss", "Methods › Loss", "Methods › Loss"]
    assert [c.page for c in chunks] == [None, None, 3, 4]
    assert chunks[1].link("S1") == "[[n#Loss|S1]]"


def test_page_link_when_no_heading():
    c = chunk_note(note("## Page 7\n\nSome extracted PDF text that is long enough to index.", "paper", "source"))[0]
    assert c.page == 7 and c.link("S2") == "[[paper#Page 7|S2]]"


def test_equations_and_code_are_never_split():
    eq = "$$\n" + " + ".join(f"a_{i} x_{i}" for i in range(60)) + "\n$$"
    code = "```python\n" + "\n".join(f"x{i} = {i}" for i in range(40)) + "\n```"
    body = f"Intro sentence about the model and its loss function.\n\n{eq}\n\n{code}\n\nClosing sentence about results."
    chunks = chunk_note(note(body), size=150, overlap=60)
    joined = [c.text for c in chunks]
    assert any(eq in t for t in joined) and any(code in t for t in joined)
    for t in joined:
        assert t.count("$$") % 2 == 0 and t.count("```") % 2 == 0


def test_overlap_repeats_whole_paragraphs_and_filters_boilerplate():
    paras = [f"Paragraph {i} says something specific about topic {i} in detail." for i in range(8)]
    chunks = chunk_note(note("\n\n".join(paras)), size=180, overlap=80)
    assert len(chunks) > 2
    first_para_of_second = chunks[1].text.split("\n\n")[0]
    assert first_para_of_second in chunks[0].text                       # overlap = whole paragraph
    src = note("> [!warning] Immutable source\n> Do not edit this text.\n\nReal content of the paper "
               "begins here with enough words.", type_="source")
    assert all("Immutable" not in c.text for c in chunk_note(src))
    assert chunk_note(note("## My notes\n_x_")) == []                   # too short to index


# ------------------------------------------------------------------ index
@pytest.fixture()
def small_vault(tmp_path):
    v = Vault(tmp_path)
    v.write("atom", "Reporting rate is not identifiable", {}, "With unknown reporting fraction the initial "
            "infected count cannot be separated from early data. Identifiability fails.")
    v.write("concept", "collocation point", {}, "## Intuition\n\nPoints where the PDE residual is enforced "
            "during PINN training; between them the residual is unconstrained.")
    v.write("source", "Paper on graph diffusion", {}, "## Page 1\n\nRumour spreading on networks follows a "
            "graph Laplacian diffusion.\n\n## Page 2\n\nPorous medium equations give finite propagation speed.")
    return v


def fake_embedder():
    return FakeClient(LLMProfile(provider="fake"))


def test_incremental_update_only_reembeds_changes(small_vault):
    idx = RagIndex(small_vault, fake_embedder(), "hash", chunk_chars=400)
    first = idx.update()
    assert first.added == 3 and first.embedded == first.chunks > 0
    assert idx.update().embedded == 0                                    # nothing changed
    small_vault.write("idea", "Laplacian PINN", {}, "Add a graph Laplacian residual to the PINN loss.")
    third = idx.update()
    assert third.added == 1 and third.embedded == 1
    (small_vault.root / "50-ideas" / "laplacian-pinn.md").unlink()
    fourth = idx.update()
    assert fourth.removed == 1 and fourth.embedded == 0
    assert all(c.slug != "laplacian-pinn" for c in idx.chunks)
    reopened = RagIndex(small_vault, fake_embedder(), "hash", chunk_chars=400)  # persisted on disk
    assert len(reopened.chunks) == len(idx.chunks) and reopened.update().embedded == 0
    resized = RagIndex(small_vault, fake_embedder(), "hash", chunk_chars=300)  # settings changed
    assert resized.update().added == 3


def test_embedding_failure_falls_back_to_keywords_then_recovers(small_vault):
    class Broken(FakeClient):
        def embed(self, texts):
            raise LLMError("server down")
    idx = RagIndex(small_vault, Broken(LLMProfile(provider="fake")), "hash")
    stats = idx.update()
    assert "server down" in stats.embedding_error and stats.chunks > 0
    assert idx.search("identifiability reporting", k=1)[0].chunk.slug == "reporting-rate-is-not-identifiable"
    healthy = RagIndex(small_vault, fake_embedder(), "hash")
    assert healthy.update().embedded == healthy.update().chunks == len(healthy.chunks)


def test_scope_cap_and_query_variants(small_vault):
    idx = RagIndex(small_vault, None)                                    # keyword-only RAG works too
    idx.update()
    assert {h.chunk.note_type for h in idx.search("diffusion Laplacian residual", scope="sources")} == {"source"}
    assert all(h.chunk.note_type != "source" for h in idx.search("diffusion residual", scope="wiki"))
    assert len(idx.search("propagation diffusion", k=5, max_per_note=1)) <= 3
    assert idx.search("limitaciones identificabilidad", k=3) == []      # Spanish words: no match
    hits = idx.search("limitaciones identificabilidad", k=3, queries=["identifiability reporting fraction"])
    assert hits[0].chunk.slug == "reporting-rate-is-not-identifiable"   # translation variant finds it


def test_chroma_backend(small_vault):
    pytest.importorskip("chromadb")
    idx = RagIndex(small_vault, fake_embedder(), "hash", backend="chroma")
    idx.update()
    assert idx.search("collocation residual", k=1)[0].chunk.slug == "collocation-point"
    small_vault.write("atom", "Extra", {}, "Completely new statement about remote sensing burned areas.")
    assert idx.update().embedded == 1


# ------------------------------------------------------------------ context and citations
def hits_for(texts):
    return [Hit(Chunk(f"n{i}::0", f"n{i}", f"Note {i}", "atom", "Sec", None, t), 1.0, i + 1)
            for i, t in enumerate(texts)]


def test_context_budget_clips_instead_of_dropping():
    hits = hits_for(["A. " * 2000, "Short passage about identifiability.", "Third passage here."])
    ctx = build_context(hits, max_chars=3000, max_passage_chars=1000)
    assert ctx.text.startswith("[S1] Note 0 › Sec") and "[S2]" in ctx.text
    assert ctx.clipped >= 1 and ctx.chars <= 3100 and "clipped" in ctx.report


def test_citations_are_linked_and_checked():
    hits = hits_for(["a", "b", "c"])
    c = link_citations("First claim with enough words here [S1]. Second claim with words too [S2; S3]. "
                       "Unsupported claim without any citation at all. Bad one with several more words [S7].", hits)
    assert "([[n0#Sec|S1]])" in c.text and "([[n1#Sec|S2]], [[n2#Sec|S3]])" in c.text
    assert c.cited == [1, 2, 3] and c.invalid == [7] and "S7⚠" in c.text
    assert c.groundedness == 0.5                     # invalid [S7] does not count
    assert "**S2** [[n1#Sec|Note 1 › Sec]]" in c.sources_section(hits)


# ------------------------------------------------------------------ brain integration
def test_retrieve_with_expansion_and_rerank(brain):
    brain.ingest(SOURCE)

    class Scorer(FakeClient):
        def chat(self, system, user, *, json_mode=False, tag=""):
            if tag == "rag-rerank":
                n = len(re.findall(r"^\[\d+\]", user, re.M))
                return '{"scores": [%s]}' % ", ".join(["0"] * (n - 1) + ["3"])  # only the last is relevant
            return super().chat(system, user, json_mode=json_mode, tag=tag)

    brain._factory = lambda p: Scorer(p)
    brain._clients.clear()
    plain = brain.retrieve("identifiability of the reporting fraction", k=3, rerank=False)
    reranked = brain.retrieve("identifiability of the reporting fraction", k=3, rerank=True)
    assert len(reranked) == 1 and reranked[0].chunk.id not in {h.chunk.id for h in plain[:1]} | {None}
    expanded = brain.retrieve("limitaciones de las PINN", k=4, expand=True)
    assert expanded                                                      # variants found English notes


def test_note_search_reuses_passage_vectors(brain, vault):
    brain.ingest(SOURCE)
    hits = brain.search_index().search("reporting fraction identifiability", k=2)
    assert hits and not (vault / ".rbrain" / "embeddings.json").exists()  # no second embedding cache
    assert (vault / ".rbrain" / "rag" / "vectors.npy").exists()


def test_condense_follow_up(brain):
    assert brain.condense("and on networks?", []) == "and on networks?"
    q = brain.condense("and on networks?", [("How do PINNs estimate beta?", "They learn it jointly.")])
    assert q.startswith("How could physics-informed neural networks")


def test_cli_rag_chat_and_ask(vault, capsys, monkeypatch):
    main(["--vault", str(vault), "ingest", str(SOURCE)])
    main(["--vault", str(vault), "rag", "index"])
    main(["--vault", str(vault), "rag", "stats"])
    main(["--vault", str(vault), "rag", "search", "collocation points residual", "-k", "2", "--scope", "sources"])
    main(["--vault", str(vault), "ask", "Limitations of PINNs?", "--show-context", "-k", "5"])
    out = capsys.readouterr().out
    assert "passages" in out and "[1]" in out and "groundedness" in out and "[S1]" in out
    inputs = iter(["What limits PINNs?", "/sources", "and on networks?", "/save", "/exit"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(inputs))
    main(["--vault", str(vault), "chat"])
    out = capsys.readouterr().out
    assert "searching for: How could physics-informed" in out and "70-answers/" in out
    assert load_config(vault).rag.top_k == 8


def test_brain_without_embeddings_still_answers(vault):
    cfg = load_config(vault)
    cfg.embeddings.enabled = False
    b = Brain(cfg, echo=lambda *_: None)
    b.ingest(SOURCE)
    ans = b.ask("Which parameters are not identifiable?")
    assert ans.context.hits and ans.citations.cited


# ------------------------------------------------------------------ MCP tools (no SDK needed)
def test_mcp_tools_and_install(brain, vault, monkeypatch):
    import json
    from research_brain.mcp_server import build_tools, install
    brain.ingest(SOURCE)
    tools = build_tools(brain)
    assert set(tools) == {"search_passages", "search_notes", "read_note", "vault_index", "logic_check", "refresh"}
    out = tools["search_passages"]("collocation points residual", k=2, scope="sources")
    assert out.startswith("[1]") and "cite: [[" in out
    assert "INVALID" in tools["logic_check"](["P -> Q", "Q"], "P")
    assert "Formula error" in tools["logic_check"](["P &"], "P")
    assert tools["read_note"]("no such note").startswith("No note")
    assert "passages" in tools["refresh"]()
    monkeypatch.setenv("RBRAIN_VAULT", str(vault))
    cfg = json.loads(install(None).read_text())
    assert cfg["mcpServers"]["research-brain"]["args"][-1] == "mcp"
