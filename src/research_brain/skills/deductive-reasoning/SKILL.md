---
name: deductive-reasoning
description: Deductive logical reasoning. Use when the user wants to check whether a conclusion follows from premises, build a step-by-step derivation, find hidden assumptions, or separate validity from soundness - for a proof sketch, a paper's argument, or a claim in the second brain. Triggers - "does this follow", "is this argument valid", "derive", "prove from these notes", "what am I assuming".
---

# Deductive logical reasoning

## Purpose
Decide whether a conclusion **follows necessarily** from a set of premises, show the
derivation step by step, expose the **hidden assumptions** a derivation silently needs,
and keep **validity** (the form is correct) apart from **soundness** (the premises are
true). The work is split between you and a symbolic engine: you formalise the argument in
propositional logic; SymPy then decides validity mechanically. If the argument is invalid,
you receive a counter-model and propose the minimal missing premise
(generate → verify → revise).

## Method
1. **Restate the goal** precisely; fix the meaning and scope of every term.
2. **Extract the premises.** Split compound premises into simple ones. Keep each premise's
   source note when it has one.
3. **Build a dictionary** of propositional atoms: short mnemonic identifiers (`Res0`, `BC`,
   `Uniq`, `Equal`) each mapped to *one* precise sentence. The same sentence always gets
   the same atom; express negation with `~`, never with a new atom ("not unique" is `~Uniq`).
4. **Formalise** every premise and the conclusion with this syntax only:
   `~` not, `&` and, `|` or, `->` implies, `<->` iff, parentheses. Example:
   `Res0 & BC -> Solves`.
5. **Derive.** Number the steps. Each step gives a formula, a sentence, the rule used and
   the premises/steps it comes from. Name standard rules: modus ponens, modus tollens,
   hypothetical syllogism, disjunctive syllogism, conjunction introduction/elimination,
   contraposition, constructive dilemma, De Morgan, case analysis, reductio ad absurdum.
6. **Hidden assumptions.** If a step needs something that is not a premise, do not use it
   silently: list it under `hidden_assumptions`. These are the most valuable output.
7. **Soundness.** Judge each premise as `supported`, `questionable` or `unsupported`, using
   the related vault notes. Watch for domain and quantifier slips: "the residual is zero
   *at the collocation points*" is not "the residual is zero *on the domain*".
8. **Avoid the classic fallacies**: affirming the consequent, denying the antecedent,
   equivocation (one word, two meanings), illicit generalisation, circularity.
9. **Know the limits.** Propositional logic cannot express "for all" and "there exists".
   When the argument hinges on quantifiers, encode the quantified statements as separate
   atoms and flag the issue in `hidden_assumptions`.

## Quality checklist
- Every formula uses only atoms defined in the dictionary and the allowed connectives.
- The conclusion formula says exactly what the goal says — not something weaker.
- Each derivation step follows from the cited lines by the named rule.
- Hidden assumptions are listed, not smuggled into the premises.
- The soundness assessment is honest even when the argument is valid.

## In Claude Code
1. Resolve premise notes (`[[slug]]`) and read them; use `rbrain search` for related evidence.
2. Formalise as in the Method, then verify mechanically — do not trust your own derivation:
   `rbrain logic check "Res0 & BC -> Solves" "Solves & Uniq -> Equal" "Res0" "BC" --goal "Equal"`
3. If the verdict is INVALID, read the counter-model, propose the minimal missing premise,
   add it and re-run the check (at most two rounds). If VACUOUS, the premises contradict
   each other: find the conflicting pair.
4. Write `60-arguments/<slug>.md` from `90-templates/argument.md`: frontmatter
   `type: argument`, `verdict: valid|invalid|vacuous`, `repaired`, `summary`; body with the
   dictionary, premises table (with sources), derivation, the symbolic verdict, hidden
   assumptions and the soundness section.
5. Run `rbrain index` and append `## [YYYY-MM-DD HH:MM] deduce | <goal>` to `log.md`.
