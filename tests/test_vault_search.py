import datetime as dt

from research_brain.config import LLMProfile
from research_brain.llm import FakeClient
from research_brain.search import BM25, SearchIndex, tokenize
from research_brain.vault import Vault, parse_note, slugify, truncate, wikilinks


def test_slugify_folds_accents_and_cuts_at_word_boundary():
    assert slugify("Ecuación de difusión (PINNs)") == "ecuacion-de-difusion-pinns"
    long = slugify("word " * 40, max_len=30)
    assert len(long) <= 30 and not long.endswith("-") and long.endswith("word")


def test_truncate_never_leaves_unbalanced_dollar():
    out = truncate("The loss $\\mathcal{L} = a + b$ and $u_\\theta(t)$ continues for a while", 45)
    assert out.count("$") % 2 == 0 and out.endswith("…")


def test_wikilinks_parsing():
    text = "See [[a-note]], [[b|Alias]], [[c#Heading]] and [[d^block|x]]."
    assert wikilinks(text) == {"a-note", "b", "c", "d"}


def test_write_read_roundtrip_and_dates(tmp_path):
    v = Vault(tmp_path)
    note = v.write("atom", "A claim about $\\beta$", {"kind": "claim", "summary": "s"}, "# A claim\n\nBody")
    assert note.path.parent.name == "30-atoms"
    meta, body = parse_note(note.path.read_text())
    assert meta["type"] == "atom" and meta["kind"] == "claim"
    assert isinstance(meta["created"], dt.date)          # written unquoted, Dataview-friendly
    assert body.startswith("# A claim")


def test_my_notes_section_is_preserved(tmp_path):
    v = Vault(tmp_path)
    n = v.write("concept", "entropy", {}, "# Entropy\n\nv1\n\n## My notes\nmine")
    v.write("concept", "entropy", {}, "# Entropy\n\nv2 regenerated", path=n.path)
    text = n.path.read_text()
    assert "v2 regenerated" in text and "## My notes\nmine" in text


def test_unique_slugs_across_folders(tmp_path):
    v = Vault(tmp_path)
    a = v.write("atom", "Same title", {}, "x")
    b = v.write("idea", "Same title", {}, "y")
    assert a.slug == "same-title" and b.slug == "same-title-2"
    assert v.resolve("[[same-title-2]]").path == b.path
    assert v.resolve("Same title").path == a.path


def test_bm25_ranks_relevant_document_first():
    docs = [tokenize(t) for t in ["heat equation diffusion", "neural network training",
                                  "physics informed neural network for the heat equation"]]
    scores = BM25(docs).scores(tokenize("physics informed heat"))
    assert scores.argmax() == 2


def test_hybrid_search_with_embeddings(tmp_path):
    v = Vault(tmp_path)
    v.write("atom", "PINNs enforce residuals at collocation points", {}, "collocation residual")
    v.write("atom", "Graph Laplacian couples nodes", {}, "graph laplacian network")
    index = SearchIndex(v, FakeClient(LLMProfile(provider="fake")), "hash")
    hits = index.search("collocation residual", k=1)
    assert hits[0][0].slug == "pinns-enforce-residuals-at-collocation-points"
    assert (tmp_path / ".rbrain" / "embeddings.json").exists()
