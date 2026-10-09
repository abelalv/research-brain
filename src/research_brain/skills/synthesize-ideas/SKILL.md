---
name: synthesize-ideas
description: Constructing new ideas. Use when the user wants new research ideas, hypotheses or project directions by combining notes in the second brain - transferring a method, drawing an analogy, inverting a problem, relaxing an assumption or resolving a contradiction. Triggers - "combine these notes", "new idea", "generate hypotheses", "what could I do with", "research directions".
---

# Constructing new ideas

## Purpose
Turn creativity into a **reproducible procedure**: apply explicit *operators* to existing
atomic notes and concepts, and return a few testable, scored, well-motivated ideas that
are linked to their parents. This draws on combinatorial creativity (new ideas as novel
combinations of existing ones), structure-mapping analogy (transfer relations, not
surface features) and Pólya's problem-solving heuristics (vary the problem).

## Method
1. **Map each parent note**: its core claim, the assumptions it rests on, and what it
   cannot do (its limits). Ideas usually live in the limits.
2. **Look for tensions and gaps** between parents: two notes that disagree, a method that
   solves a problem adjacent to an open question, an assumption one note needs and
   another note shows to be false.
3. **Apply operators.** Use at least two different operators across the ideas:
   - `method-transfer` — use the method of A on the problem of B
     (e.g. PINN training for coefficient estimation in rumour-spreading models);
   - `analogy` — map the *relational structure* of A onto B (heat diffusion on a domain ↔
     information diffusion on a graph); say where the analogy breaks;
   - `combination` — merge two mechanisms into one model (GNN spatial coupling + physics loss);
   - `inversion` — swap what is known and unknown (forward problem → inverse problem);
   - `assumption-relaxation` — drop or weaken an assumption (constant → spatially varying
     coefficients; deterministic → stochastic);
   - `generalization` / `specialization` — widen the scope, or find a tractable special case;
   - `scale-shift` — change the scale (individual → population, local → network, daily → seasonal);
   - `contradiction-resolution` / `gap-filling` — propose what would reconcile two notes or
     fill a documented gap.
4. **Generate more than you return.** Internally produce about three times the number of
   ideas requested, then keep the best ones by the scores below, preferring diversity.
5. **Write each idea as a testable proposition** ("Adding a graph-Laplacian residual to the
   PINN loss reduces coefficient-estimation error under sparse sensors"), not as a topic.
6. **Novelty check.** State what is new relative to the parents and to the existing ideas
   listed in the request. Be honest: if the idea probably exists in the literature, say so
   and say what search would confirm it. Never invent references.
7. **Test.** Give the smallest decisive experiment, proof or dataset that could refute it.
8. **Score** each idea from 1 to 5:
   - *novelty*: 1 = restates a parent … 5 = no parent implies it;
   - *feasibility*: 1 = needs unavailable data or new mathematics … 5 = doable in weeks
     with existing tools;
   - *impact*: 1 = marginal … 5 = would change practice or settle an open question.
   Calibrate: most ideas are 2–4; reserve 5 for exceptional cases.
9. **Next step**: one concrete action that takes at most a day.

## Quality checklist
- Each idea is falsifiable and more than "A and B together".
- Each idea cites only the parent slugs it really uses.
- Assumptions and risks are explicit (data availability, identifiability, compute).
- Ideas are diverse: different operators, not three variations of one idea.
- No fabricated citations, datasets or results.

## In Claude Code
1. Read the parent notes the user names; if they give a theme, run
   `rbrain search "<theme>" --type atom --type concept -k 8` and read the top hits.
2. Check existing ideas with `rbrain search "<theme>" --type idea` to avoid duplicates.
3. Apply the Method and write each idea to `50-ideas/<slug>.md` using
   `90-templates/idea.md`: frontmatter `type: idea`, `status: seed`, `operator`, `novelty`,
   `feasibility`, `impact`, `parents: ["[[slug]]", ...]`, `summary`.
4. Run `rbrain index` and append `## [YYYY-MM-DD HH:MM] synthesize | <theme>` to `log.md`.
5. Offer to turn the best idea into a deductive argument (`deductive-reasoning`) or an
   experiment plan.
