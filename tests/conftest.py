"""Shared fixtures: a fresh vault wired to the fake provider and the example responses."""

from pathlib import Path

import pytest

from research_brain import bootstrap
from research_brain.config import load_config
from research_brain.workflows import Brain

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"

FAKE_CONFIG = """
[vault]
language = "English"
[llm]
provider = "fake"
model = "demo"
[embeddings]
enabled = true
provider = "fake"
model = "hash"
"""


@pytest.fixture()
def vault(tmp_path, monkeypatch):
    monkeypatch.setattr(bootstrap, "USER_CONFIG", tmp_path / "default_vault")
    monkeypatch.setenv("RBRAIN_FAKE_DIR", str(EXAMPLES / "fake-responses"))
    root = tmp_path / "vault"
    bootstrap.init_vault(root, set_default=False)
    (root / ".rbrain" / "config.toml").write_text(FAKE_CONFIG, encoding="utf-8")
    return root


@pytest.fixture()
def brain(vault):
    return Brain(load_config(vault), echo=lambda *_: None)
