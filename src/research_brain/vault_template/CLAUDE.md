# Research Brain — schema for the AI agent

This vault is a **research second brain** maintained by an LLM, following the LLM-wiki
pattern: raw sources → an interlinked wiki the agent writes → this schema file. The human
curates sources, asks questions and decides what matters; the agent does the reading,
summarising, linking, filing and bookkeeping. `AGENTS.md` is an identical copy for Codex,
OpenCode and other agents.

## Owner and style
- The owner is a mathematics professor and researcher (SciML, PINNs, GNNs, differential
  equations, mathematical epidemiology, remote sensing). Assume graduate-level mathematics.
- Write notes in English, clear and academic but approachable. Use LaTeX: inline `$...$`,
  display `$$...$$`. Define every symbol.
- Be honest about uncertainty. Never invent papers, authors, numbers or results.

## Layout
| Folder | Type | Owner | Content |
|---|---|---|---|
| `00-inbox/` | — | human | drop zone for new sources and quick thoughts |
| `10-sources/` | `source` | **immutable** | extracted text of each source; binaries in `10-sources/assets/` |
| `20-summaries/` | `summary` | agent | one page per source: question, methods, results, limits, concepts |
| `30-atoms/` | `atom` | agent | atomic ideas: one claim / definition / assumption / result per note |
| `40-concepts/` | `concept` | agent (+ human "## My notes") | layered explanations of concepts |
| `50-ideas/` | `idea` | agent + human | new hypotheses with parents, operator, test, scores |
| `60-arguments/` | `argument` | agent | deductive arguments with symbolic verdict |
| `70-answers/` | `answer` | agent | answers to questions, filed back so they compound |
| `90-templates/` | — | human | note templates |

Special files: `index.md` (catalogue — read it first), `log.md` (append-only timeline),
`Dashboard.md` (Dataview queries), `lint-report.md` (latest health check).

## Conventions
- **File names** are slugs (`physics-informed-neural-network.md`) and are unique across the
  vault, so `[[slug]]` links are unambiguous. Display text: `[[slug|Readable Title]]`.
- **Frontmatter** on every note: `title`, `type`, `created`, `updated`, `summary`, `tags`,
  plus type-specific fields — atoms: `kind`, `confidence`, `source`; ideas: `status`
  (`seed` → `growing` → `evergreen` → `dropped`), `operator`, `novelty`, `feasibility`,
  `impact`, `parents`; arguments: `verdict`, `repaired`; concepts: `level`, `aliases`.
- **Atom titles are declarative sentences**; concept titles are canonical noun phrases
  (singular, lowercase unless proper nouns).
- **Never edit `10-sources/`.** Never delete a human-written "## My notes" section.
- **Contradictions** are recorded, not resolved silently: add a
  `> [!warning] Contradiction` callout in both notes, linking each other.
- Every write ends with: run `rbrain index`, then append to `log.md` a line
  `## [YYYY-MM-DD HH:MM] <operation> | <title>` followed by `- [[created-note]]` lines.

## Skills (in `.claude/skills/`)
| Skill | Use it to |
|---|---|
| `ingest-source` | add a source: immutable copy, summary page, then segmentation |
| `segment-idea` | split any text into typed atomic ideas linked to evidence |
| `synthesize-ideas` | construct new testable ideas with explicit creative operators |
| `explain-concept` | write layered concept pages (intuition → formal → example → self-test) |
| `deductive-reasoning` | formalise an argument, verify validity symbolically, find hidden assumptions |
| `query-brain` | RAG: answer from passages of notes and sources with citations, report gaps |
| `lint-brain` | health check and research agenda |

## Deterministic tools you can call (the `rbrain` CLI, or the MCP tools if enabled)
Use these instead of guessing — they are fast and exact:
- `rbrain rag search "<question>" [-k 8] [--scope all|wiki|sources] [--full]` — passage-level
  RAG over notes AND the full text of sources; prints `[[note#Heading]]` links to cite.
  MCP equivalent: `search_passages`. Use `--scope sources` for exact numbers and quotes.
- `rbrain search "<query>" [-k 8] [--type atom|concept|idea|summary|argument]` — whole notes
  (MCP: `search_notes`); `read_note` / `vault_index` read a note or the catalogue.
- `rbrain logic check "<P1>" "<P2>" ... --goal "<C>"` — propositional validity with counter-model.
  Syntax: `~ & | -> <->` and parentheses. **Always run it before calling an argument valid.**
- `rbrain logic table "<formula>"` — truth table.
- `rbrain index` — regenerate `index.md` after writing notes (MCP: `refresh`, which also
  re-indexes the new notes for retrieval).
- `rbrain lint` — broken links, orphans, duplicates, unsourced claims → `lint-report.md`.

## Typical session
1. Read `index.md`; retrieve passages for the request (`rbrain rag search` / `search_passages`)
   and open the notes they come from.
2. Apply the matching skill; link every new note to its parents, sources and concepts.
3. When an answer, comparison or derivation is valuable, propose filing it as a note.
4. Finish with `rbrain index` and a `log.md` entry; tell the user which notes changed.
