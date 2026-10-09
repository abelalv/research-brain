---
name: ingest-source
description: Ingest a new source (paper, article, chapter, lecture notes, transcript) into the second brain - keep an immutable copy, write a summary page, identify concepts and open questions, then segment it into atomic ideas. Triggers - "ingest", "add this paper", "process this source", "file this article", a file dropped in 00-inbox.
---

# Ingest a source

## Purpose
Compile a source into the wiki **once**, so later questions build on it instead of
re-reading it (Karpathy's LLM-wiki pattern). The summary page is the entry point to the
source: what it asks, how it answers, what it finds, where it is weak, and which concepts
it touches. Atomic ideas are produced afterwards by `segment-idea`.

## Method
1. Identify the **research question** and the **main contribution** in two or three
   sentences (`summary`). Write it so it can be read without the source.
2. **Key points**: the 3–7 most important claims or findings, each one sentence.
3. **Methods**: data, models, algorithms, proofs or experimental design actually used.
4. **Results**: concrete outcomes, with numbers and conditions when the text gives them.
5. **Limitations**: those the authors state and those you can see (assumptions,
   small samples, missing baselines, identifiability). Mark your own observations with
   "(reader's note)".
6. **Concepts**: 3–10 canonical concept names that deserve their own page
   (singular, lowercase unless proper nouns).
7. **Open questions**: what remains unanswered or would be the natural next study.
8. **Tags**: 3–8 short topical tags.
9. If the text is truncated (marked "[... omitted ...]"), say so instead of guessing what
   the omitted part contains.

## Quality checklist
- Numbers and claims match the source exactly; no invented results.
- The summary states the contribution, not the topic.
- Limitations include at least one reader's note when warranted.

## In Claude Code
1. Move or copy the file from `00-inbox/` into `10-sources/` (PDFs into
   `10-sources/assets/`); never edit files in `10-sources/` afterwards.
2. Write `20-summaries/summary-<slug>.md` from `90-templates/summary.md` with
   frontmatter `type: summary`, `source: "[[<source-slug>]]"`, `summary`, `tags`.
3. Discuss the key takeaways with the user before filing, if they are present.
4. Run the `segment-idea` skill on the source and list the atoms under "## Atomic ideas".
5. Update existing concept pages that the source changes (add evidence, flag contradictions
   with a `> [!warning] Contradiction` callout linking both notes).
6. Run `rbrain index`; append `## [YYYY-MM-DD HH:MM] ingest | <title>` to `log.md`.
