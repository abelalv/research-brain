import json

import pytest

from research_brain.config import LLMProfile
from research_brain.llm import FakeClient, LLMError, chat_json, extract_json, repair_latex_escapes, strip_think


def test_json_inside_prose_and_fences():
    assert extract_json('Sure!\n```json\n{"a": 1}\n```\nDone.') == {"a": 1}
    assert extract_json('Here: {"a": {"b": [1, 2]}} and more text') == {"a": {"b": [1, 2]}}


def test_think_blocks_are_removed():
    assert strip_think("<think>let me think {x}</think>\nanswer") == "answer"
    assert extract_json('<think>{"wrong": true}</think>{"right": true}') == {"right": True}


def test_trailing_commas_are_tolerated():
    assert extract_json('{"a": [1, 2,], "b": 3,}') == {"a": [1, 2], "b": 3}


@pytest.mark.parametrize("latex", [r"\frac{a}{b}", r"\nabla u", r"\theta", r"\rho", r"\beta", r"\text{res}",
                                   r"\alpha", r"\partial_t u", r"\mathcal{L}", r"\| u \|", r"\underline{x}"])
def test_single_backslash_latex_survives(latex):
    raw = '{"m": "$' + latex + '$"}'
    assert extract_json(raw)["m"] == f"${latex}$"


def test_real_json_escapes_are_kept():
    raw = r'{"m": "line one\nline two\ttab \"quoted\" \\frac done é"}'
    assert extract_json(raw)["m"] == 'line one\nline two\ttab "quoted" \\frac done é'


def test_newline_followed_by_word_is_not_mistaken_for_latex():
    assert extract_json(r'{"m": "a\nnumerical b"}')["m"] == "a\nnumerical b"


def test_repair_is_idempotent_on_valid_doubled_latex():
    raw = json.dumps({"m": r"$\frac{1}{2}\nabla$"})
    assert repair_latex_escapes(raw) == raw


def test_chat_json_retries_once_then_fails():
    client = FakeClient(LLMProfile(provider="fake"), {"t": "no json here", "t.retry": '{"ok": 1}'})
    assert chat_json(client, "sys", "user", tag="t") == {"ok": 1}
    bad = FakeClient(LLMProfile(provider="fake"), {"t": "nope", "t.retry": "still nope"})
    with pytest.raises(LLMError):
        chat_json(bad, "sys", "user", tag="t")
