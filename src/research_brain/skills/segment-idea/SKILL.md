---
name: segment-idea
description: Idea segmentation. Use when the user wants to break a paper, chapter, draft, lecture or brainstorm into atomic, linkable ideas (claims, definitions, assumptions, methods, results, open questions) for the second brain. Triggers - "segment", "atomize", "break this into ideas", "extract the claims", "make atomic notes".
---

# Idea segmentation

## Purpose
Decompose a text into **atomic ideas**: one idea per note, each understandable on its own,
typed by its role in the argument, and linked to its evidence. Atomic notes are the unit
of a research second brain: they can be searched, linked across sources, recombined into
new ideas (see `synthesize-ideas`) and used as premises (see `deductive-reasoning`).
The method combines Zettelkasten atomicity with argument mapping (Toulmin: claim,
evidence, warrant/assumption).

## Method
1. **Read the whole text once** and identify the author's main question and thesis. Put
   them in `main_question` and `structure`.
2. **Find the units by their function**, using exactly one `kind` per atom:
   - `definition` — fixes the meaning of a term or symbol;
   - `assumption` — something taken as given (modelling choice, hypothesis of a theorem);
   - `claim` — an assertion the author argues for;
   - `method` — how something is done (algorithm, experimental design, loss function);
   - `result` — an outcome obtained (theorem, measured number, observed effect);
   - `evidence` — data or observation offered in support of a claim;
   - `question` — an open problem, limitation or future work, phrased as a question;
   - `hypothesis` — a conjecture not yet tested.
3. **Apply the atomicity test** to every candidate:
   - it has one subject and one predicate; if "and"/"but" joins two assertions that could
     be independently true or false, split it;
   - it is **decontextualised**: no "this", "it", "the proposed method", "Eq. (3)";
     name the thing and define every symbol you keep;
   - someone who never read the source understands it.
4. **Title = declarative sentence of at most 12 words** that states the idea
   ("PINN residuals are enforced only at collocation points"), never a topic label
   ("PINN residuals").
5. **Preserve epistemic status.** Hedged language ("suggests", "may", "in our experiments")
   lowers `confidence`. Distinguish the author's own claims from claims they cite.
   Never upgrade a correlation into a cause or a numerical observation into a theorem.
6. **Evidence and locator.** Give a short quote (at most 40 words) or a close paraphrase,
   plus the section, page or equation number when visible.
7. **Dependencies.** In `depends_on`, list the indices (0-based, within your list) of atoms
   this one relies on: results depend on methods and assumptions, claims on definitions.
   This turns the list into an argument graph.
8. **Concepts.** For each atom, name 1–4 concepts that deserve their own page, using
   canonical, singular, lowercase names ("inverse problem", "collocation point").
9. **Prioritise.** Central and novel ideas first; skip boilerplate, acknowledgements and
   generic background that any textbook states.
10. **Mathematics.** Keep equations in LaTeX and define each symbol in the same atom.

## Quality checklist
- Every atom passes the stand-alone test and has exactly one kind.
- No two atoms say the same thing; no atom is a list of several ideas.
- Assumptions are extracted explicitly — they are what later reasoning most often misses.
- Limitations and open problems appear as `question` atoms.
- Nothing is invented: if the text does not say it, it is not an atom.
- Respect the maximum number of atoms given in the request.

## In Claude Code
When running inside the vault with Claude Code (not through the `rbrain` CLI):
1. Read the source (from `10-sources/` or the file the user gives) and apply the Method.
2. For each atom create `30-atoms/<slug>.md` following `90-templates/atom.md`: frontmatter
   `type: atom`, `kind`, `confidence`, `source: "[[<source-slug>]]"`, `summary` (the
   statement), `tags`; body with the statement, an evidence callout, and links to the
   dependencies (`[[slug]]`) and concepts (`[[concept-slug|concept name]]`).
3. Before writing, run `rbrain search "<statement>" -k 3` and link the related notes it finds,
   so new atoms connect to existing knowledge across sources.
4. Add the list of new atoms under "## Atomic ideas" in the source's summary page.
5. Run `rbrain index` and append `## [YYYY-MM-DD HH:MM] segment | <title>` to `log.md`.
