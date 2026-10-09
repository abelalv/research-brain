# Workflows

The skills are only useful if they become habits. These routines are designed so that the
vault grows a little every day and turns into research output every few weeks. Times are
indicative; on a CPU-only machine, run ingestion in the background.

## Daily (10–15 minutes)

1. **Capture without friction.** New PDFs go to `00-inbox/`; quick thoughts become a note
   from the *fleeting thought* template (Obsidian: *Templates: Insert template*). Web
   articles: Obsidian Web Clipper into `00-inbox/`.
2. **Ingest one source**, the one you would otherwise read superficially:
   ```bash
   rbrain ingest 00-inbox/paper.pdf
   ```
3. **Review the atoms** for five minutes in Obsidian: delete trivial ones, fix confidence,
   make sure the key assumptions are there. This review is where *you* learn the paper.
4. If a thought from the day is worth keeping, segment it:
   `rbrain segment 00-inbox/thought.md`.
5. When a question comes up while reading or writing, ask the vault instead of searching
   by hand: `rbrain ask "..."` or keep `rbrain chat` open; `/save` the answers worth keeping.

## Weekly (45 minutes)

1. **Health check and agenda:** `rbrain lint --llm`, then read `lint-report.md`.
2. **Fill vocabulary gaps:** explain the two or three most-linked missing concepts
   (`rbrain explain "collocation point"`). Write your own version under `## My notes`.
3. **Synthesis session:** pick a `question` atom and a method or result from a different
   cluster; run `rbrain synthesize "[[question]]" "[[method]]" -q "<focus>"`.
4. **Triage ideas** in `Dashboard.md`: run a real literature search on the top ones, then set
   `status` to `growing`, `dropped` (with the reason in the note) or leave as `seed`.
5. **Commit:** `git add -A && git commit -m "week NN"` in the vault.

## From reading to a paper

1. **State the research question** as a `question` atom (or write it in a fleeting note and
   segment it). Link it from a project note in `00-inbox/` or a folder of your choice.
2. **Build the corpus:** ingest 10–20 key sources. Run `rbrain lint`; explain the core
   concepts so the vocabulary is precise and shared across notes.
3. **Map the field:** `rbrain ask "What do my notes say about <topic>? Where do they disagree?" --save`.
   Saved answers with citations are the raw material of a related-work section.
4. **Generate candidates:** `rbrain synthesize -q "<research question>" --n 5`; select one
   idea after a literature check; set it to `growing`.
5. **Argue the main claim:** `rbrain deduce -p "[[atom]]" ... -g "<main claim>"`. Every
   premise the repair loop adds is an assumption the paper must state; every
   *questionable* premise in the soundness section is a limitation or an experiment.
6. **Write:** the argument note gives the skeleton of the theory section, the idea's *test*
   gives the experiments, the cited answers give related work. Draft in your LaTeX
   project; cite sources from `10-sources/` (their frontmatter keeps the original path).

## Teaching

- **Concept pages for a course:** `rbrain explain "eigenvalue" --level undergraduate`.
  The misconceptions section is a list of what to address in class; the self-test questions
  are quiz items.
- **Logic and proof:** `rbrain logic table "(P -> Q) <-> (~P | Q)"` and
  `rbrain logic check "P -> Q" "Q" --goal "P"` (affirming the consequent, with
  counter-model) work without any model.
- **Students' arguments:** paste a student's proof sketch into `deduce` to show where an
  unstated assumption is used.

## Reviewing a thesis proposal or a manuscript

1. Ingest it (`--no-segment` if you only want the summary), then segment the sections that
   carry the argument.
2. Check that a `question`, a `method` and an expected `result` exist and are connected
   (`depends_on` links).
3. Run `deduce` on the main claim with the atoms as premises; hidden assumptions and
   questionable premises become your review comments.

## CLI or Claude Code?

Use the **CLI with a local model** for bulk and private work: ingestion, segmentation, daily
explanations, search. Use **agent mode** (`rbrain agent`, i.e. Claude Code in the vault)
for work that benefits from a strong model reading many notes and editing several files:
weekly synthesis across clusters, contradiction hunting during lint, restructuring a
concept area, drafting a related-work section with citations. `rbrain agent --backend
deepseek` runs the same agent on DeepSeek. Both modes follow the same skills and
conventions, so you can mix them freely. A middle path is
`rbrain --profile claude synthesize ...`: one command with Claude through your
subscription, everything else local (see [`CONNECTIONS.md`](CONNECTIONS.md)).

## Maintenance

- `rbrain index --embed` after editing many notes by hand.
- Changing the embedding model invalidates the cache automatically (it is keyed by model).
- To update the skills after upgrading the package: `rbrain init <vault> --force` overwrites
  the template files and skills (your notes are never touched; customised skills will be
  replaced, so keep a copy or use `--link-skills` while developing).
- Back up the vault with git, and keep the sources' originals (`10-sources/assets/`).
