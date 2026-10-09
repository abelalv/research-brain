---
name: explain-concept
description: Explaining concepts. Use when the user wants a concept explained, a concept page written or regenerated in the second brain, or study / teaching material for a term (intuition, formal definition, worked example, misconceptions, self-test). Triggers - "explain", "what is", "concept page", "teach me", "fill the missing page".
---

# Explaining concepts

## Purpose
Write a **layered concept page** that builds real understanding and can be reused for
teaching. The layers follow well-tested learning principles: the Feynman technique (say
it simply to expose gaps), dual coding (intuition *and* formalism), worked examples
(reduce cognitive load before abstraction), conceptual change (name and correct the
typical misconceptions) and retrieval practice (end with self-test questions).

## Method
1. **One line**: what the concept is and what it is for, in one sentence.
2. **Intuition**: a picture or analogy with no jargon. Say explicitly where the analogy
   breaks down.
3. **Formal definition**: precise statement in LaTeX. Define every symbol, give the
   setting (spaces, assumptions, regularity) and the conditions under which it holds.
4. **Worked example**: the smallest non-trivial example, computed step by step, with
   numbers when possible. Show the computation, not just the answer.
5. **Misconceptions**: 2–4 typical errors (from students or from practice), each with
   a short correction.
6. **Prerequisites**: the concepts needed to understand this one, by canonical name
   (they become links; missing ones are reported by `lint` as pages to write).
7. **Related concepts**: siblings, generalisations and contrasts ("collocation" vs
   "Galerkin projection").
8. **Self-test**: 3–5 questions of increasing difficulty — recall, application, transfer —
   each with a concise answer.
9. **Adapt to the level** in the request: `intro` (no formulas beyond algebra, more
   intuition), `undergraduate`, `graduate` (default: full formalism), `expert` (focus on
   subtleties, edge cases and open issues).
10. **Ground it in the vault**: use the notes provided as context when they are relevant
    and list their slugs in `vault_links`. Do not cite notes that are not related.

## Quality checklist
- The intuition and the formal definition describe the same object.
- Every symbol in the formal definition is defined.
- The worked example can be checked by hand.
- Misconceptions are real and specific, not strawmen.
- Self-test answers are correct and short.
- No invented references or historical claims you are unsure of.

## In Claude Code
1. Run `rbrain search "<concept>" -k 6` and read the hits for context.
2. If `40-concepts/<slug>.md` exists, update it in place and **keep the "## My notes"
   section untouched** (it belongs to the user). Otherwise create it from
   `90-templates/concept.md` with frontmatter `type: concept`, `level`, `summary`
   (the one-line definition), `aliases`, `tags`.
3. Use the concept name exactly as written in existing `[[links]]` so that the page
   resolves links that were previously broken.
4. Self-test questions use foldable callouts: `> [!question]- Question` followed by the
   answer on the next `>` lines.
5. Run `rbrain index` and append `## [YYYY-MM-DD HH:MM] explain | <concept>` to `log.md`.
