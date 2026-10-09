"""Deterministic Markdown rendering of skill outputs into Obsidian notes.

The LLM supplies content (JSON); these functions supply structure: headings, callouts,
wikilinks and checklists. Keeping layout in code means every note of a given type looks
the same regardless of which model wrote it, which keeps Dataview queries and the graph
view reliable.
"""

from __future__ import annotations

from typing import Iterable

from .logic import ArgumentCheck
from .vault import link, slugify


def _bullets(items: Iterable[str], empty: str = "_None recorded._") -> str:
    items = [str(i).strip() for i in items if str(i).strip()]
    return "\n".join(f"- {i}" for i in items) if items else empty


def concept_links(names: Iterable[str]) -> str:
    """Concept names become [[slug|Name]] links; unresolved ones show up in lint as gaps."""
    names = [n for n in names if str(n).strip()]
    return ", ".join(link(slugify(n), n) for n in names) if names else "_None_"


def _quote_block(text: str) -> str:
    return "\n".join(f"> {line}" if line else ">" for line in str(text).splitlines())


# ------------------------------------------------------------------------- summary
def summary_body(d: dict, source_link: str) -> str:
    return f"""# {d.get('title', 'Summary')}

> [!abstract] Summary
{_quote_block(d.get('summary', ''))}

**Source:** {source_link}

## Key points
{_bullets(d.get('key_points', []))}

## Methods
{_bullets(d.get('methods', []))}

## Results
{_bullets(d.get('results', []))}

## Limitations
{_bullets(d.get('limitations', []))}

## Concepts
{concept_links(d.get('concepts', []))}

## Open questions
{_bullets(d.get('open_questions', []))}

## Atomic ideas
_Filled in by idea segmentation._
"""


# ------------------------------------------------------------------------- atom
def atom_body(a: dict, source_link: str | None, related: list[str], depends: list[str]) -> str:
    evidence = str(a.get("evidence", "")).strip()
    locator = str(a.get("locator", "")).strip()
    ev = ""
    if evidence:
        where = ", ".join(x for x in (source_link, locator) if x)
        ev = "\n> [!quote] Evidence\n" + _quote_block(evidence) + (f"\n> — {where}" if where else "") + "\n"
    return f"""# {a.get('title', 'Atom')}

{a.get('statement', '').strip()}
{ev}
**Kind:** {a.get('kind', 'claim')} · **Confidence:** {a.get('confidence', 'medium')}

## Links
- Source: {source_link or '_own thought_'}
- Depends on: {', '.join(depends) if depends else '_—_'}
- Concepts: {concept_links(a.get('concepts', []))}
- Related notes: {', '.join(related) if related else '_none found yet_'}
"""


# ------------------------------------------------------------------------- idea
def idea_body(idea: dict, parent_links: list[str], similar: list[str]) -> str:
    s = idea.get("scores", {}) or {}
    return f"""# {idea.get('title', 'Idea')}

> [!tip] Hypothesis
{_quote_block(idea.get('statement', ''))}

## Genealogy
- **Operator:** {idea.get('operator', 'combination')}
- **Parents:** {', '.join(parent_links) if parent_links else '_—_'}

## Rationale
{idea.get('rationale', '').strip()}

## Why it might be new
{idea.get('novelty_check', '').strip()}

**Similar notes already in the vault:** {', '.join(similar) if similar else '_none found_'}

## How to test or falsify it
{idea.get('test', '').strip()}

## Assumptions
{_bullets(idea.get('assumptions', []))}

## Risks
{_bullets(idea.get('risks', []))}

## Scores
| Novelty | Feasibility | Impact |
|:-:|:-:|:-:|
| {s.get('novelty', '?')}/5 | {s.get('feasibility', '?')}/5 | {s.get('impact', '?')}/5 |

## Next step
- [ ] {idea.get('next_step', 'Define the smallest experiment that could refute this.')}
"""


# ------------------------------------------------------------------------- concept
def concept_body(c: dict, vault_links: list[str]) -> str:
    mis = c.get("misconceptions", []) or []
    mis_md = "\n".join(
        f"- **Misconception:** {m.get('misconception', '')}\n  **Correction:** {m.get('correction', '')}"
        for m in mis if isinstance(m, dict)
    ) or "_None recorded._"
    qs = c.get("self_test", []) or []
    qs_md = "\n\n".join(
        f"> [!question]- {q.get('question', '')}\n{_quote_block(q.get('answer', ''))}"
        for q in qs if isinstance(q, dict)
    ) or "_None._"
    name = str(c.get("title", "Concept"))
    return f"""# {name[:1].upper() + name[1:]}

> [!info] In one line
{_quote_block(c.get('one_line', ''))}

## Intuition
{c.get('intuition', '').strip()}

## Formal definition
{c.get('formal_definition', '').strip()}

## Worked example
{c.get('worked_example', '').strip()}

## Common misconceptions
{mis_md}

## Prerequisites
{concept_links(c.get('prerequisites', []))}

## Related concepts
{concept_links(c.get('related', []))}

## In this vault
{', '.join(vault_links) if vault_links else '_Nothing linked yet._'}

## Self-test
_Click a question to reveal the answer (retrieval practice)._

{qs_md}

{'## My notes'}
_This section is yours: it is preserved when the concept page is regenerated._
"""


# ------------------------------------------------------------------------- argument
def argument_body(d: dict, check: ArgumentCheck, added: list[dict], history: list[str]) -> str:
    dictionary = d.get("dictionary", {}) or {}
    dict_md = "\n".join(f"| `{k}` | {v} |" for k, v in dictionary.items()) or "| — | — |"
    prem_rows = []
    for i, p in enumerate(d.get("premises", []) or [], start=1):
        src = p.get("source")
        src_md = link(src) if src else "_stated_"
        prem_rows.append(f"| P{i} | {p.get('text', '')} | `{p.get('formula', '')}` | {src_md} |")
    for j, p in enumerate(added, start=len(prem_rows) + 1):
        prem_rows.append(f"| P{j} ⚠ | {p.get('text', '')} _(added: {p.get('justification', '')})_ "
                         f"| `{p.get('formula', '')}` | _missing premise, plausibility {p.get('plausibility', '?')}_ |")
    steps = []
    for s in d.get("derivation", []) or []:
        frm = ", ".join(map(str, s.get("from", []) or []))
        steps.append(f"{s.get('step', '?')}. `{s.get('formula', '')}` — {s.get('statement', '')} "
                     f"_({s.get('rule', '')}{'; from ' + frm if frm else ''})_")
    sound = d.get("soundness", []) or []
    sound_md = "\n".join(f"- **{x.get('premise', '')}** — {x.get('assessment', '')}: {x.get('comment', '')}"
                         for x in sound if isinstance(x, dict)) or "_Not assessed._"
    concl = d.get("conclusion", {}) or {}
    callout = "success" if check.valid and check.premises_consistent else "failure"
    return f"""# {d.get('title') or concl.get('text', 'Argument')}

> [!{callout}] Verdict
{_quote_block(check.to_markdown())}

## Goal
{concl.get('text', '')}  → `{concl.get('formula', '')}`

## Formalisation dictionary
| Atom | Meaning |
|---|---|
{dict_md}

## Premises
| # | Statement | Formula | Source |
|---|---|---|---|
{chr(10).join(prem_rows) if prem_rows else '| — | — | — | — |'}

## Derivation
{chr(10).join(steps) if steps else '_No derivation produced._'}

## Hidden assumptions
{_bullets(d.get('hidden_assumptions', []))}

## Soundness (are the premises actually true?)
{sound_md}

> [!note] Validity vs soundness
> The symbolic check only guarantees **validity** (the form is correct). Whether the
> conclusion is **true** depends on the premises, which the section above assesses.

## Model's informal verdict
{d.get('informal_verdict', '').strip()}

## Verification history
{_bullets(history)}
"""
