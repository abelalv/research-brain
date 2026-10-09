"""Symbolic checking of deductive arguments (propositional logic, via SymPy).

The LLM is good at turning prose into premises and proposing derivations, and bad at
*guaranteeing* that a conclusion follows. So the deductive-reasoning skill splits the work:

1. the LLM formalises premises and conclusion into propositional formulas;
2. this module decides validity mechanically:  premises |= conclusion  iff
   (P1 & ... & Pn & ~C) is unsatisfiable;
3. if the argument is invalid, the counter-model is fed back to the LLM, which proposes the
   missing premise(s); the check is repeated (generate -> verify -> revise).

Formula syntax (ASCII or Unicode)::

    ~P  !P  ¬P  not P          negation
    P & Q   P ∧ Q   P and Q     conjunction
    P | Q   P ∨ Q   P or Q      disjunction
    P -> Q  P => Q  P → Q       implication (right associative)
    P <-> Q P <=> Q P ↔ Q       equivalence
    true / false / ⊤ / ⊥        constants

Atoms are identifiers such as ``P``, ``Residual0`` or ``well_posed``.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field

from sympy import Symbol, satisfiable
from sympy.logic.boolalg import And, Boolean, Equivalent, Implies, Not, Or, false, true


class FormulaError(ValueError):
    """Raised for malformed formulas."""


_TOKEN_RE = re.compile(
    r"\s*(<->|<=>|↔|->|=>|→|&&|&|∧|\|\||\||∨|~|¬|!|\(|\)|⊤|⊥|[A-Za-z_][A-Za-z0-9_']*)"
)
_WORDS = {"and": "&", "or": "|", "not": "~", "implies": "->", "iff": "<->",
          "true": "TRUE", "false": "FALSE"}
_NORMALISE = {"<=>": "<->", "↔": "<->", "=>": "->", "→": "->", "&&": "&", "∧": "&",
              "||": "|", "∨": "|", "¬": "~", "!": "~", "⊤": "TRUE", "⊥": "FALSE"}


def _tokenize(text: str) -> list[str]:
    tokens, pos = [], 0
    text = text.strip()
    while pos < len(text):
        m = _TOKEN_RE.match(text, pos)
        if not m or m.end() == pos:
            raise FormulaError(f"Unexpected character at position {pos}: {text[pos:pos + 10]!r}")
        tok = m.group(1)
        tok = _NORMALISE.get(tok, tok)
        tok = _WORDS.get(tok.lower(), tok)  # keyword operators: and, or, not, implies, iff
        tokens.append(tok)
        pos = m.end()
        while pos < len(text) and text[pos].isspace():
            pos += 1
    return tokens


class _Parser:
    """Recursive-descent parser. Precedence: ~  >  &  >  |  >  ->  >  <->."""

    def __init__(self, tokens: list[str]):
        self.tokens, self.i = tokens, 0
        self.symbols: dict[str, Symbol] = {}

    def peek(self) -> str | None:
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def take(self, expected: str | None = None) -> str:
        tok = self.peek()
        if tok is None or (expected and tok != expected):
            raise FormulaError(f"Expected {expected or 'a token'} but found {tok!r}")
        self.i += 1
        return tok

    def parse(self) -> Boolean:
        expr = self.iff()
        if self.peek() is not None:
            raise FormulaError(f"Unexpected token {self.peek()!r}")
        return expr

    def iff(self) -> Boolean:
        left = self.imp()
        while self.peek() == "<->":
            self.take()
            left = Equivalent(left, self.imp())
        return left

    def imp(self) -> Boolean:
        left = self.disj()
        if self.peek() == "->":
            self.take()
            return Implies(left, self.imp())
        return left

    def disj(self) -> Boolean:
        left = self.conj()
        while self.peek() == "|":
            self.take()
            left = Or(left, self.conj())
        return left

    def conj(self) -> Boolean:
        left = self.unary()
        while self.peek() == "&":
            self.take()
            left = And(left, self.unary())
        return left

    def unary(self) -> Boolean:
        if self.peek() == "~":
            self.take()
            return Not(self.unary())
        return self.atom()

    def atom(self) -> Boolean:
        tok = self.take()
        if tok == "(":
            expr = self.iff()
            self.take(")")
            return expr
        if tok == "TRUE":
            return true
        if tok == "FALSE":
            return false
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_']*", tok):
            return self.symbols.setdefault(tok, Symbol(tok))
        raise FormulaError(f"Unexpected token {tok!r}")


def parse_formula(text: str) -> Boolean:
    """Parse a propositional formula into a SymPy Boolean expression."""
    if not text or not text.strip():
        raise FormulaError("Empty formula")
    return _Parser(_tokenize(text)).parse()


def _model(assignment) -> dict[str, bool]:
    if not assignment:
        return {}
    return {str(k): bool(v) for k, v in assignment.items() if isinstance(k, Symbol)}


@dataclass
class ArgumentCheck:
    """Result of checking ``premises |= conclusion``."""

    premises: list[str]
    conclusion: str
    valid: bool
    premises_consistent: bool
    counterexample: dict[str, bool] | None = None
    essential: list[int] = field(default_factory=list)   # indices of premises the proof needs
    atoms: list[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        if not self.premises_consistent:
            return "VACUOUS: the premises contradict each other, so anything follows (ex falso)."
        if self.valid:
            return "VALID: the conclusion follows from the premises."
        return "INVALID: there is an assignment making every premise true and the conclusion false."

    def to_markdown(self) -> str:
        lines = [f"**Symbolic verdict:** {self.verdict}"]
        if self.counterexample:
            ce = ", ".join(f"`{k}={'T' if v else 'F'}`" for k, v in sorted(self.counterexample.items()))
            lines.append(f"**Counter-model:** {ce}")
        if self.valid and self.essential:
            used = ", ".join(f"P{i + 1}" for i in self.essential)
            lines.append(f"**Premises the conclusion actually depends on:** {used}")
        return "\n\n".join(lines)


def check_argument(premises: list[str], conclusion: str) -> ArgumentCheck:
    """Decide whether ``conclusion`` is a logical consequence of ``premises``."""
    parsed = [parse_formula(p) for p in premises]
    goal = parse_formula(conclusion)
    conj = And(*parsed) if parsed else true
    consistent = satisfiable(conj) is not False
    witness = satisfiable(And(conj, Not(goal)))
    valid = witness is False
    atoms = sorted({str(s) for s in conj.free_symbols | goal.free_symbols})
    essential: list[int] = []
    if valid and consistent:
        for i in range(len(parsed)):
            rest = And(*(parsed[:i] + parsed[i + 1:])) if len(parsed) > 1 else true
            if satisfiable(And(rest, Not(goal))) is not False:
                essential.append(i)
    return ArgumentCheck(
        premises=premises, conclusion=conclusion, valid=valid, premises_consistent=consistent,
        counterexample=None if valid else _model(witness), essential=essential, atoms=atoms,
    )


def truth_table(formula: str, max_atoms: int = 6) -> str:
    """Markdown truth table, handy for teaching and for sanity checks."""
    expr = parse_formula(formula)
    atoms = sorted(expr.free_symbols, key=str)
    if len(atoms) > max_atoms:
        raise FormulaError(f"{len(atoms)} atoms; truth tables are limited to {max_atoms}.")
    header = "| " + " | ".join(map(str, atoms)) + f" | {formula} |"
    sep = "|" + "---|" * (len(atoms) + 1)
    rows = []
    for values in itertools.product([True, False], repeat=len(atoms)):
        val = bool(expr.subs(dict(zip(atoms, values))))
        cells = ["T" if v else "F" for v in values] + ["**T**" if val else "F"]
        rows.append("| " + " | ".join(cells) + " |")
    kind = classify(formula)
    return "\n".join([header, sep, *rows, "", f"_{kind}_"])


def classify(formula: str) -> str:
    """Tautology, contradiction or contingent."""
    expr = parse_formula(formula)
    if satisfiable(Not(expr)) is False:
        return "Tautology (true under every assignment)"
    if satisfiable(expr) is False:
        return "Contradiction (false under every assignment)"
    return "Contingent (true under some assignments, false under others)"
