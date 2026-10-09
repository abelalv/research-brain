import pytest

from research_brain.logic import FormulaError, check_argument, classify, parse_formula, truth_table


def test_modus_ponens_is_valid():
    r = check_argument(["P -> Q", "P"], "Q")
    assert r.valid and r.premises_consistent and r.counterexample is None
    assert r.essential == [0, 1]


def test_affirming_the_consequent_is_invalid():
    r = check_argument(["P -> Q", "Q"], "P")
    assert not r.valid
    assert r.counterexample == {"P": False, "Q": True}


def test_missing_uniqueness_premise_is_detected():
    premises = ["Res0 & IC -> Solves", "Solves & Uniq -> Equal", "Res0", "IC"]
    r = check_argument(premises, "Equal")
    assert not r.valid and r.counterexample["Uniq"] is False
    assert check_argument(premises + ["Uniq"], "Equal").valid


def test_inconsistent_premises_are_vacuous():
    r = check_argument(["P", "~P"], "Q")
    assert not r.premises_consistent
    assert "VACUOUS" in r.verdict


def test_unused_premise_is_not_essential():
    r = check_argument(["P -> Q", "P", "R"], "Q")
    assert r.valid and r.essential == [0, 1]


@pytest.mark.parametrize("text", ["¬P ∨ Q", "not P or Q", "!P || Q", "(P → Q)", "P implies Q"])
def test_alternative_syntaxes_are_equivalent(text):
    assert check_argument([text], "P -> Q").valid
    assert check_argument(["P -> Q"], text).valid


def test_implication_is_right_associative():
    # P -> Q -> R  ==  P -> (Q -> R)
    assert check_argument(["P -> Q -> R"], "P -> (Q -> R)").valid
    assert not check_argument(["P -> Q -> R"], "(P -> Q) -> R").valid


def test_atoms_named_like_sympy_constants_work():
    # E, I, S, N, Q would clash with SymPy names if sympify were used.
    assert check_argument(["S & I -> E", "S", "I"], "E").valid


def test_contraposition_tautology_and_table():
    assert classify("(P -> Q) <-> (~Q -> ~P)").startswith("Tautology")
    assert classify("P & ~P").startswith("Contradiction")
    table = truth_table("P -> Q")
    assert table.count("\n| ") == 4 and "Contingent" in table


@pytest.mark.parametrize("bad", ["", "P &", "(P | Q", "P $ Q", "-> P"])
def test_malformed_formulas_raise(bad):
    with pytest.raises(FormulaError):
        parse_formula(bad)
