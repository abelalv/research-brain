# The research skills and how to apply them

A second brain for research is not a place to store notes; it is a **pipeline that turns
reading into publishable thinking**. Ordinary note-taking stops at capture. This system
adds four cognitive operations that research actually consists of, and gives each one to
the LLM as an explicit, inspectable procedure (an Agent Skill), while you keep the
judgement.

```
capture ──► SEGMENT ──► connect ──► EXPLAIN ──► CONSTRUCT ──► DEDUCE ──► write
   ▲          (atoms)    (search,     (concepts)   (ideas)      (arguments)   │
   │                     links)                                               │
   └───────────── ASK / LINT: gaps become questions, questions become sources ┘
```

The division of labour follows Karpathy's LLM-wiki idea: the model does the bookkeeping
that makes humans abandon knowledge bases (summarising, linking, filing, keeping the index
current), and you do what only a researcher can do — choose sources, decide what is
important, judge whether an idea is worth months of work. Each skill below is described
in four parts: **what it is**, **why it works**, **how the system implements it**, and
**how you should apply it** in your own practice.

---

## 1. Idea segmentation

### What it is
Splitting a text — a paper, a chapter, a thesis proposal, your own brainstorm — into
**atomic ideas**: one idea per note, understandable without the source, typed by the role it
plays in the argument, and linked to its evidence.

### Why it works
A paper is a *linear* argument, but knowledge is used *non-linearly*: you need "the
assumption behind Theorem 2" next to "the counterexample from another paper", not the whole
two PDFs. Atomicity (the core rule of the Zettelkasten method) makes ideas recombinable.
Typing atoms by function comes from argument mapping (Toulmin's claim / data / warrant):
when every atom is labelled *definition*, *assumption*, *claim*, *method*, *result*,
*evidence*, *question* or *hypothesis*, the hidden structure of the text becomes visible —
in particular the **assumptions**, which are what later reasoning most often forgets.

### How the system implements it
`segment-idea` asks the model for at most $k$ atoms per chunk (default 10), each with a
declarative title, a self-contained statement (symbols defined, no "this" or "Eq. (3)"),
a kind, a confidence level that preserves the author's hedging, a short quote with its
locator, 1–4 concept names, and `depends_on` indices. The code then:

1. writes each atom to `30-atoms/` with a unique slug;
2. turns `depends_on` into links, so the atoms of one source form an **argument graph**;
3. searches the existing vault for each statement and links the top hits, so the new atoms
   connect to knowledge from **other** sources (this is where cross-paper insight starts);
4. turns concept names into `[[links]]` — the ones without a page appear in `lint` as
   missing pages, i.e. your vocabulary to-do list.

**The atomicity test**, which the skill applies and you should apply when reviewing:

| Bad atom | Why | Good atom |
|---|---|---|
| "PINN residuals" | topic, not an idea | "PINN residuals are enforced only at collocation points" |
| "The method works well and is fast" | two claims, vague | "PINNs recover a constant $\beta$ within 5% from 20 noisy observations" + a separate atom for speed, if supported |
| "As shown in Eq. 3, it converges" | depends on the source | "Gradient descent on $\mathcal{L}(\theta,\beta)$ converged in all 10 synthetic runs with $\lambda_r = 1$" |

### How to apply it
Segment **every source you will cite** (via `rbrain ingest`) and **every brainstorm you want to
keep** (`rbrain segment 00-inbox/note.md`). Afterwards spend five minutes reviewing: delete
trivial atoms, raise or lower confidence, and above all check that the assumptions were
extracted — if the paper's key assumption is missing, add it by hand from the template.
Use the kinds deliberately: `question` atoms are your future research topics, `assumption`
atoms are what your arguments must state, `result` atoms are what you cite. When you
review a student's thesis proposal, segmenting it is the fastest way to see whether the
research question, the method and the expected result actually connect.

**Pitfalls.** Over-segmentation (twenty trivial atoms per page) makes the graph noisy —
keep the cap low and prefer central ideas. Small models sometimes upgrade hedged claims
("suggests") into facts; the confidence field is there so you can catch it.

---

## 2. Constructing new ideas

### What it is
Producing **new, testable research ideas** by applying explicit creative *operators* to
existing notes, then scoring and recording them with their genealogy.

### Why it works
Most research ideas are recombinations: a method from one field applied to a problem from
another, an assumption dropped, a forward problem inverted. Cognitive science calls this
combinatorial creativity, and analogy research (Gentner's structure mapping) shows that
*good* analogies transfer **relations** (e.g. "diffusion operator acting on a state"), not
surface features ("both involve networks"). Pólya's heuristics — vary the problem,
specialise, generalise, work backwards — are the same operators applied to mathematics.
Making the operator explicit turns "be creative" into a procedure you can repeat and audit.

### How the system implements it
`synthesize-ideas` receives parent notes (the ones you name plus search hits for your
focus), the ideas already in the vault (to avoid repeats), and must return ideas that each
state a **testable proposition**, name the operator and parents, explain the mechanism,
check novelty honestly ("this probably exists; search X to confirm"), give the smallest
refuting test, list assumptions and risks, and score novelty, feasibility and impact on
calibrated 1–5 scales. The code keeps only parent slugs that really exist, links similar
existing ideas, and files everything in `50-ideas/` with `status: seed`.

| Operator | Move | Example in SciML / mathematical biology |
|---|---|---|
| method-transfer | method of A on problem of B | PINN training used for coefficient estimation in rumour-spreading PDEs |
| analogy | map relational structure | heat diffusion on a domain ↔ information diffusion on a graph (graph Laplacian) |
| combination | merge mechanisms | GNN spatial coupling + physics residual in one loss |
| inversion | swap known and unknown | forward SIR simulation → infer the mobility network |
| assumption-relaxation | weaken a hypothesis | constant $\beta$ → $\beta(x,t)$; linear diffusion → porous medium $u_t = \Delta(u^m)$ |
| generalization / specialization | widen scope / find tractable case | any graph → trees, where the inverse problem may be explicit |
| scale-shift | change scale | individual contacts → metapopulations; daily → seasonal |
| contradiction-resolution / gap-filling | reconcile or fill | two atoms disagree on identifiability → design the experiment that decides |

### How to apply it
Run synthesis **weekly**, not after every paper: ideas need a critical mass of atoms to
combine. Good inputs are a `question` atom (what is open) plus a `method` or `result` atom
from a different source (what could answer it): `rbrain synthesize "[[open-question]]"
"[[method-from-other-paper]]" -q "focus"`. Then do three things the model cannot: run a
real literature search on each promising idea (LLMs systematically overestimate novelty),
decide which ones deserve time, and promote them by editing `status` from `seed` to
`growing` (being worked on), `evergreen` (mature, written up) or `dropped` (keep it — a
dropped idea with its reason is knowledge). The Dashboard ranks ideas by the sum of
scores; your status field is what turns the list into a research pipeline. The best idea
of the week is a natural input to the next skill: can its central claim actually be
argued?

**Pitfalls.** Idea lists that are variations of one idea (the skill requires different
operators); "A + B" ideas with no mechanism (reject anything without a refuting test);
fabricated citations in the novelty check (the skill forbids them; still verify).

---

## 3. Explaining concepts

### What it is
Writing a **layered concept page** that builds understanding: one-line definition, intuition
(with its limits), formal definition, worked example, misconceptions, prerequisites and
related concepts, connections to your notes, and self-test questions.

### Why it works
Each layer is a well-established learning principle. The **Feynman technique**: if you
cannot explain it simply, you have found a gap. **Dual coding**: understanding is robust
when an intuitive picture and a formal statement describe the same object. **Worked
examples** reduce cognitive load before abstraction (cognitive load theory).
**Misconception-based teaching** (conceptual change): naming the typical error is more
effective than restating the right idea. **Retrieval practice**: answering questions
strengthens memory far more than re-reading — so every page ends with foldable questions.

### How the system implements it
`explain-concept` searches the vault for related notes and grounds the explanation in them
(only slugs that really exist are kept as links), adapts to the requested level
(`intro`, `undergraduate`, `graduate`, `expert`), and writes `40-concepts/<concept>.md`
using the **same slug as the links that already point to it** — so explaining a concept
repairs the broken links reported by `lint`. Regenerating a page keeps your `## My notes`
section.

### How to apply it
Use it in three loops. **(a) Vocabulary loop:** after each `rbrain lint`, explain the two or
three most-linked missing concepts; your concept layer grows exactly where your reading
needs it. **(b) Feynman loop:** before generating a page, write your own two-paragraph
explanation in a fleeting note; then generate the page and compare. Where they differ is
what you did not understand (or where the model is wrong — check formal definitions against
a textbook). Keep your corrected version under `## My notes`. **(c) Teaching loop:**
generate at `undergraduate` level for your courses (linear algebra, modelling, deep learning)
and reuse the self-test questions as quiz items; the misconceptions section is a ready-made
list of what to address in class. Open the self-test callouts a few days later without
looking at the page — that is spaced retrieval.

**Pitfalls.** Polished but wrong formalism (verify every definition you will reuse in a
paper); analogies presented without their breaking point (the skill requires it).

---

## 4. Deductive logical reasoning

### What it is
Deciding whether a conclusion **follows necessarily** from premises, displaying the
derivation, exposing **hidden assumptions**, and keeping **validity** (the form is correct)
apart from **soundness** (the premises are true).

### Why it works — and why the LLM is not trusted alone
For premises $\Gamma = \{P_1,\dots,P_n\}$ and conclusion $C$,

$$
\Gamma \models C \iff P_1 \wedge \dots \wedge P_n \wedge \neg C \ \text{is unsatisfiable}.
$$

Deciding this is mechanical, and language models are unreliable at it: they produce fluent
derivations that silently use an unstated premise or commit classic fallacies (affirming the
consequent, equivocation). The skill therefore uses a neuro-symbolic split. The model does
what it is good at — translating prose into a dictionary of propositional atoms and
formulas, proposing a derivation, judging whether premises are supported by the vault. SymPy
does what it is good at — deciding satisfiability exactly. When the argument is invalid,
SymPy returns a **counter-model** (an assignment making every premise true and the
conclusion false); the model must propose the *minimal* premise that rules it out, and the
check is repeated (generate → verify → revise, at most `--rounds` times). If the premises
are inconsistent, the verdict is **vacuous**: anything follows (*ex falso quodlibet*), which
usually reveals two notes that contradict each other.

### Worked example (from the demo)
Claim: *the trained PINN equals the true SIR trajectory*. Dictionary: `Res0` (residual zero
on the interval), `IC` (initial conditions hold), `Solves`, `Uniq` (the IVP has a unique
solution), `Equal`. Premises: `Res0 & IC -> Solves`, `Solves & Uniq -> Equal`, `Res0`, `IC`.

- Round 0: **invalid**, counter-model `Uniq = F`. The informal proof had used uniqueness
  without stating it.
- Repair: add `Uniq`, justified by Picard–Lindelöf (the SIR vector field is polynomial, hence
  locally Lipschitz; the invariant simplex $S+I+R=N$ gives global existence). Round 1:
  **valid**; all five premises are essential.
- Soundness: P3 (`Res0`) is **questionable** — training only makes the residual small at
  collocation points. So the honest conclusion is weaker: the network is close to the true
  trajectory where the residual was checked. That sentence is exactly what a referee would
  ask you to write.

### How to apply it
Run it **before you write the main claim of a paper or proposal**: feed the claim and the
atoms it rests on (`-p "[[atom]]"`), and treat every added premise as a sentence that must
appear in the paper's assumptions. Use it when **reviewing** a thesis or a manuscript:
segment the argument, then check whether the conclusion follows from what is actually
stated. Use `rbrain logic` alone in **teaching** (truth tables, validity, tautologies) — no
model needed. Always read the soundness section: a valid argument from a questionable
premise is the most common failure in applied mathematics papers.

**Limits.** Propositional logic cannot express quantifiers ("for all $x \in \Omega$",
"there exists $\varepsilon$"), so the skill encodes quantified statements as atoms and flags
quantifier slips in the hidden assumptions. For arguments that hinge on quantifiers or
arithmetic, the natural extension is an SMT solver (Z3) or a proof assistant (Lean); the
checker in `logic.py` is deliberately small so it can be swapped.

---

## 5. Supporting skills

**`ingest-source`** compiles a source into the wiki once: immutable copy in `10-sources/`,
summary page (question, methods, results with numbers, limitations including "reader's
notes", concepts, open questions), then segmentation. **`query-brain`** is the RAG skill: it
answers from passages retrieved from your notes *and* the full text of your sources, with a
citation behind every factual sentence that links to the exact section or PDF page; it labels
general knowledge that is not in the vault, reports contradictions and ends with *Gaps*;
`--save` files the answer so your questions also compound (details in [`RAG.md`](RAG.md)). **`lint-brain`** computes missing pages, orphans, duplicates,
unsourced claims and invalid arguments, and (with `--llm`) proposes specific research
questions and kinds of sources to find — the vault's gaps become your reading list.

---

## 6. Choosing the right skill

| You want to… | Use |
|---|---|
| file a paper so you never re-read it from scratch | `rbrain ingest` |
| see the structure and assumptions of an argument | `segment-idea` |
| understand or teach a term; fill a missing page | `explain-concept` |
| find a research direction or a paper idea | `synthesize-ideas` |
| test whether a claim follows and what it assumes | `deductive-reasoning` |
| recall what you know, with sources | `query-brain` |
| decide what to read or work on next | `lint-brain` |

A typical chain for a paper: **ingest** 10–20 sources → **lint** → **explain** the core
concepts → **synthesize** across clusters → pick one idea → **deduce** its main claim from
atoms → the argument note (premises, hidden assumptions, soundness) becomes the skeleton of
the paper's theory section, and the idea's "test" becomes the experiments section.

---

## 7. Writing your own skill

Copy a folder in `<vault>/.claude/skills/`, rename it, and edit. A skill needs:

- `SKILL.md` with frontmatter (`name`, `description` — the description is how Claude Code
  decides when to use it, so write the trigger phrases there) and the sections
  `## Purpose`, `## Method`, `## Quality checklist` (sent to any model by the CLI) and
  `## In Claude Code` (only for Claude Code);
- `schema.json`, the JSON contract the CLI asks for.

To run a new skill from the CLI, add a method to `workflows.py` (gather context →
`self.run_skill(name, prompt)` → render → `self.finish(...)`) and a renderer in
`render.py`; the existing workflows are short templates to copy. Natural next skills for a
research brain: `critique-paper` (structured peer review against a checklist),
`proof-sketch` (plan a proof and list lemmas), `experiment-plan` (turn an idea's test into
a reproducible protocol), `related-work` (draft a related-work section from summaries with
citations).
