Analyse the argument below deductively.

1. Restate the conclusion precisely.
2. List the premises (split compound ones). Mark each premise's source if given.
3. Build a dictionary of propositional atoms with short names (e.g. Res0, BC, Uniq), one precise sentence each; negations use ~.
4. Formalise premises and conclusion using only: ~ & | -> <-> and parentheses.
5. Give a numbered derivation; each step names its rule (modus ponens, modus tollens, hypothetical syllogism, disjunctive syllogism, contraposition, case analysis, reductio…) and the lines it uses.
6. List hidden assumptions that a step needs but no premise provides.
7. Judge soundness: for each premise, supported / questionable / unsupported, and why. Watch for quantifier and domain slips ("at sampled points" vs "everywhere").
8. Name any fallacy (affirming the consequent, denying the antecedent, equivocation, circularity).

End with a code block containing the command to verify validity mechanically:

```
rbrain logic check "<P1>" "<P2>" ... --goal "<C>"
```

Argument:
{}
