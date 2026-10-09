---
name: lint-brain
description: Health-check the second brain - missing concept pages, orphan notes, duplicates, unsourced claims, invalid arguments, contradictions - and propose the next research questions and sources. Triggers - "lint", "health check", "what is missing in my vault", "weekly review", "what should I read next".
---

# Lint the second brain

## Purpose
Keep the knowledge base healthy as it grows, and turn its gaps into a research agenda.
The deterministic part (broken links, orphans, duplicates, unsourced claims, stale index)
is computed by `rbrain lint`; this skill interprets the findings and proposes **questions
worth investigating** and **sources worth finding**.

## Method
1. Read the lint findings and the index.
2. Group the missing pages by theme; the most-linked missing concepts are the priority.
3. Look for **gaps between clusters**: two well-developed areas with no ideas or arguments
   connecting them are candidates for synthesis.
4. Look for **weak foundations**: many notes resting on low-confidence or unsourced atoms,
   or on arguments that are not valid.
5. Propose 3–7 research questions. Each must be specific, answerable, and linked to the
   slugs it comes from (`related`). Explain in one sentence why it matters.
6. Suggest kinds of sources to look for (e.g. "a review of identifiability for SIR
   inverse problems"), never invented titles or authors.

## Quality checklist
- Questions are specific and connected to existing notes.
- No invented references.

## In Claude Code
1. Run `rbrain lint` and read `lint-report.md` and `index.md`.
2. Also scan for contradictions between notes on the same concept (the CLI cannot detect
   these): flag each with a `> [!warning] Contradiction` callout in both notes.
3. Append the questions to `lint-report.md` as a checklist; offer to run `explain-concept`
   for the top missing pages and `synthesize-ideas` across disconnected clusters.
4. Append `## [YYYY-MM-DD HH:MM] lint | <summary>` to `log.md`.
