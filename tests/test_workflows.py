"""End-to-end tests of every workflow with the fake provider and the example responses."""

from conftest import EXAMPLES

from research_brain.bootstrap import doctor
from research_brain.cli import main
from research_brain.config import load_config
from research_brain.skills import load_skill

SOURCE = EXAMPLES / "sources" / "pinn-sir-inverse-problem.md"
A_METHOD = "a-pinn-treats-the-transmission-rate-as-a-trainable-parameter"
A_QUESTION = "can-pinns-estimate-spatially-varying-transmission-on-contact-networks"


def test_init_creates_layout_and_skills(vault):
    for folder in ["00-inbox", "10-sources", "30-atoms", "60-arguments", ".obsidian", "copilot-custom-prompts"]:
        assert (vault / folder).is_dir()
    assert (vault / "CLAUDE.md").read_text() == (vault / "AGENTS.md").read_text()
    assert (vault / ".claude" / "skills" / "deductive-reasoning" / "SKILL.md").exists()


def test_skill_prompt_contains_method_and_schema(vault):
    skill = load_skill("segment-idea", vault)
    prompt = skill.system_prompt("English")
    assert "atomicity test" in prompt and '"depends_on"' in prompt
    assert "In Claude Code" not in prompt  # Claude-Code-only section is not sent to local models


def test_ingest_summary_and_atoms(brain, vault):
    notes = brain.ingest(SOURCE)
    types = [n.type for n in notes]
    assert types.count("source") == 1 and types.count("summary") == 1 and types.count("atom") == 7
    summary = next(n for n in notes if n.type == "summary")
    assert f"[[{A_METHOD}|" in summary.path.read_text()           # atoms listed in the summary
    method = (vault / "30-atoms" / f"{A_METHOD}.md").read_text()
    assert "Depends on: [[the-sir-model" in method                  # argument graph from depends_on
    assert "\\mathcal{L}_{\\text{res}}" in method                   # LaTeX intact
    assert "ingest | Physics-informed" in (vault / "log.md").read_text()
    assert brain.ingest(SOURCE)[0].type == "source"                 # duplicate source is skipped


def test_explain_repairs_latex_and_filters_links(brain, vault):
    brain.ingest(SOURCE)
    note = brain.explain("physics-informed neural network")
    text = note.path.read_text()
    assert "\\frac{1}{N_d}" in text and "\\nabla" not in text and "\\mathcal{N}" in text
    assert "\x0c" not in text                                          # no form feed from "\frac"
    assert "not-a-real-note" not in text                               # hallucinated link dropped
    assert "> [!question]- " in text and "## My notes" in text


def test_synthesize_ideas_with_parents(brain):
    brain.ingest(SOURCE)
    ideas = brain.synthesize([f"[[{A_METHOD}]]", f"[[{A_QUESTION}]]"], query="graph neural networks for spatial spread")
    assert len(ideas) == 3
    assert {i.meta["operator"] for i in ideas} == {"combination", "inversion", "gap-filling"}
    assert all(A_METHOD in str(i.meta["parents"]) for i in ideas)
    assert ideas[0].meta["novelty"] == 3


def test_deduce_repairs_invalid_argument(brain):
    note, check = brain.deduce(["p1", "p2", "p3", "p4"], "The trained PINN equals the true SIR trajectory")
    assert check.valid and note.meta["repaired"] is True and note.meta["verdict"] == "valid"
    text = note.path.read_text()
    assert "Round 0: INVALID" in text and "added `Uniq`" in text and "P5 ⚠" in text


def test_ask_saves_answer_and_lint_finds_gaps(brain, vault):
    brain.ingest(SOURCE)
    brain.explain("physics-informed neural network")
    ans = brain.ask("What are the main limitations of PINNs?", save=True)
    assert ans.citations.cited == [1, 2, 3, 4, 5] and "|S1]]" in ans.text and "## Sources" in ans.text
    assert ans.note.path.parent.name == "70-answers" and ans.note.meta["groundedness"] > 0.5
    report = brain.lint(use_llm=True)
    assert "collocation-point" in report.broken                       # concept linked but not written
    assert "physics-informed-neural-network" not in report.broken     # filled by explain
    assert "Suggested research questions" in report.path.read_text()


def test_doctor_and_cli(vault, capsys):
    results = doctor(load_config(vault))
    assert all(ok for ok, msg in results if not msg.startswith("Claude Code"))
    main(["--vault", str(vault), "logic", "check", "P -> Q", "P", "--goal", "Q"])
    assert "VALID" in capsys.readouterr().out
    main(["--vault", str(vault), "ingest", str(SOURCE), "--no-segment"])
    main(["--vault", str(vault), "search", "transmission rate", "-k", "3"])
    out = capsys.readouterr().out
    assert "20-summaries" in out and "[summary" in out
