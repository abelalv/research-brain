---
name: query-brain
description: Answer a question from the second brain (RAG) with citations to the exact passages of notes and sources, state clearly what the vault does not cover, and file valuable answers back as notes. Triggers - "what do my notes say about", "ask my brain", "according to my vault", "summarise what I know about", "find in my papers".
---

# Query the second brain (RAG)

## Purpose
Answer from **what the vault already knows** — the researcher's own notes (summaries,
atoms, concepts, ideas, arguments) and the full text of the sources — with a citation behind
every factual statement, so the answer can be checked by clicking through. Good answers are
filed back into `70-answers/`, which is how questions make the knowledge base grow.

## Method
1. Read the numbered passages. Each starts with `[S#]`, its location (note › section, page)
   and its note type. Prefer concept pages and summaries for definitions, sources for exact
   numbers and quotes, atoms for specific claims, arguments for what has been proven, ideas
   for speculation.
2. Answer directly in the first sentence, then give the supporting detail.
3. **Cite** every factual sentence with the passage ids it relies on, in square brackets:
   `[S2]` or `[S1, S4]`. Cite only ids that appear in the passages; never invent one.
4. Keep the epistemic status visible: label ideas as hypotheses, low-confidence atoms as
   tentative, and arguments with their verdict (valid / invalid).
5. If you add background knowledge that is not in the passages, mark it explicitly as
   "(general knowledge, not in the vault)" and do not cite it.
6. If passages **contradict** each other, say so and cite both.
7. If the passages do not answer the question, say so plainly instead of guessing.
8. Finish with a short "**Gaps**" paragraph: what the vault does not cover and which note,
   source or `rbrain explain` / `rbrain synthesize` command would fill the gap.
9. If a conversation is given, use it only to understand the question; cite passages, not
   earlier answers.
10. Use Markdown and LaTeX ($...$, $$...$$). Return Markdown only (no JSON).

## Quality checklist
- Every factual sentence has a valid [S#] citation or a "general knowledge" label.
- No invented notes, papers, passages or results.
- Contradictions and gaps are reported, not smoothed over.

## In Claude Code
1. Retrieve passages with the MCP tool `search_passages` (if the research-brain MCP server is
   enabled) or with `rbrain rag search "<question>" -k 8 --full`; use `--scope sources` for
   exact facts from papers and `--scope wiki` for your synthesis. Read `index.md` for an overview.
2. Answer following the Method, but cite with the Obsidian links the tool prints
   (`[[note#Heading]]`) instead of [S#], so the answer is clickable in the vault.
3. If the answer is worth keeping, ask the user, then save it to `70-answers/<slug>.md`
   (`type: answer`, `question`, `summary`) and append `## [YYYY-MM-DD HH:MM] ask | <question>`
   to `log.md`; run `rbrain index` (or the MCP tool `refresh`).
