"""Connections: provider registry, secrets, config editing, the connect wizard, Claude Code
(subscription) client, LiteLLM client, Ollama detection/pull and the agent launcher."""

import json
import os
import stat
import sys
import threading
import tomllib
import types
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from research_brain import connect, providers
from research_brain.agent import agent_env, export_lines
from research_brain.config import LLMProfile, config_path, load_config
from research_brain.connect import Wizard, set_key, upsert_profile
from research_brain.llm import LLMError, make_client

# ------------------------------------------------------------------ mock Ollama server
MODELS = ["qwen2.5:7b-instruct-q4_K_M", "deepseek-r1:14b", "bge-m3:latest"]


class OllamaHandler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, payload):
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/version":
            self._json({"version": "0.15.2"})
        elif self.path == "/api/tags":
            self._json({"models": [{"name": m} for m in MODELS]})

    def do_POST(self):
        data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/api/pull":
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            for ev in [{"status": "pulling manifest"}, {"status": "downloading", "total": 100, "completed": 50},
                       {"status": "downloading", "total": 100, "completed": 100}, {"status": "success"}]:
                self.wfile.write((json.dumps(ev) + "\n").encode())
            MODELS.append(data["model"])
        elif self.path == "/api/chat":
            self._json({"message": {"content": "OK"}})
        elif self.path == "/api/embed":
            self._json({"embeddings": [[0.5, 0.5] for _ in data["input"]]})


@pytest.fixture()
def mock_ollama(monkeypatch):
    srv = HTTPServer(("127.0.0.1", 0), OllamaHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("OLLAMA_HOST", f"127.0.0.1:{srv.server_port}")
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


@pytest.fixture()
def secrets(tmp_path, monkeypatch):
    path = tmp_path / "secrets.env"
    monkeypatch.setattr(providers, "SECRETS_FILE", path)
    return path


# ------------------------------------------------------------------ registry and secrets
def test_registry_defaults_fill_profiles():
    p = LLMProfile(provider="deepseek")
    assert p.base_url == "https://api.deepseek.com" and p.api_key_env == "DEEPSEEK_API_KEY"
    assert p.model == "deepseek-flash" and p.json_mode is True
    assert LLMProfile(provider="claude_code").model == "sonnet"
    for name in ["openrouter", "gemini", "groq", "mistral", "openai", "together", "lmstudio"]:
        assert LLMProfile(provider=name, model="m").spec.api == "openai"
    with pytest.raises(ValueError):
        LLMProfile(provider="nope")


@pytest.mark.parametrize("host,url", [("", "http://localhost:11434"), ("192.168.1.5", "http://192.168.1.5:11434"),
                                      ("0.0.0.0:9999", "http://0.0.0.0:9999"),
                                      ("http://win-host:11434", "http://win-host:11434")])
def test_ollama_host_variants(monkeypatch, host, url):
    monkeypatch.setenv("OLLAMA_HOST", host)
    assert providers.ollama_url() == url


def test_secrets_are_private_and_env_wins(secrets, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    providers.save_secret("GROQ_API_KEY", "gsk-first")
    providers.save_secret("GROQ_API_KEY", "gsk-second")         # replaced, not duplicated
    assert stat.S_IMODE(os.stat(secrets).st_mode) == 0o600
    assert secrets.read_text().count("GROQ_API_KEY=") == 1
    monkeypatch.setenv("MISTRAL_API_KEY", "from-env")
    secrets.write_text(secrets.read_text() + "export MISTRAL_API_KEY='from-file'\n")
    providers.load_secrets()
    assert os.environ["MISTRAL_API_KEY"] == "from-env"
    assert providers.mask("sk-1234567890abcdef") == "sk-1…cdef"


# ------------------------------------------------------------------ TOML editing
BASE = """[vault]
language = "English"

[llm]
# default_profile = "x"
provider = "ollama"
model = "qwen2.5:7b-instruct-q4_K_M"

[llm.profiles.reasoning]
provider = "ollama"
model = "deepseek-r1:14b"

[skill_profiles]
deductive-reasoning = "reasoning"
# synthesize-ideas = "claude"

[embeddings]
enabled = false
"""


def test_config_editing_roundtrip():
    text = upsert_profile(BASE, "deepseek", {"provider": "deepseek", "model": "deepseek-v4-pro", "base_url": ""})
    text = set_key(text, "llm", "default_profile", "deepseek")
    text = set_key(text, "skill_profiles", "synthesize-ideas", "deepseek")
    data = tomllib.loads(text)
    assert data["llm"]["default_profile"] == "deepseek"
    assert data["llm"]["profiles"]["deepseek"] == {"provider": "deepseek", "model": "deepseek-v4-pro"}
    assert data["skill_profiles"] == {"deductive-reasoning": "reasoning", "synthesize-ideas": "deepseek"}
    text = upsert_profile(text, "deepseek", {"provider": "deepseek", "model": "deepseek-flash"})  # replace
    assert tomllib.loads(text)["llm"]["profiles"]["deepseek"]["model"] == "deepseek-flash"
    assert text.count("[llm.profiles.deepseek]") == 1
    text = set_key(text, "llm", "default_profile", None)                                       # remove
    assert "default_profile" not in tomllib.loads(text)["llm"]


# ------------------------------------------------------------------ wizard
def test_wizard_flags_only(vault, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    Wizard(vault, interactive=False, echo=lambda *_: None).run(
        provider="deepseek", model="deepseek-v4-pro", use="skills",
        skills=["synthesize-ideas", "deductive-reasoning"], test=False)
    cfg = load_config(vault)
    assert cfg.profile_for("synthesize-ideas").model == "deepseek-v4-pro"
    assert cfg.profile_for("deductive-reasoning").provider == "deepseek"
    assert cfg.llm.provider == "fake"                                 # default untouched
    assert (config_path(vault).with_suffix(".toml.bak")).exists()     # backup written


def test_wizard_interactive_asks_key_lists_models_and_sets_default(vault, secrets, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(providers, "list_models", lambda *a, **k: ["anthropic/claude-x", "deepseek/deepseek-y"])
    monkeypatch.setattr(connect, "test_profile", lambda p: (True, "answered 'OK' in 0.1 s"))
    answers = iter(["openrouter", "2", "", "1"])        # provider, model #2, default name, use=default
    out = []
    Wizard(vault, ask=lambda q: next(answers), ask_secret=lambda q: "sk-or-secret",
           echo=out.append).run()
    cfg = load_config(vault)
    assert cfg.llm.provider == "openrouter" and cfg.llm.model == "deepseek/deepseek-y"
    assert "OPENROUTER_API_KEY=sk-or-secret" in secrets.read_text()
    assert "sk-or-secret" not in config_path(vault).read_text()   # key never in the vault
    assert any("Saved profile [openrouter]" in line for line in out)


def test_wizard_failed_test_does_not_save(vault, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "x")
    monkeypatch.setattr(connect, "test_profile", lambda p: (False, "401 unauthorized"))
    before = config_path(vault).read_text()
    with pytest.raises(LLMError):
        Wizard(vault, interactive=False, echo=lambda *_: None).run(provider="groq", model="m")
    assert config_path(vault).read_text() == before


def test_wizard_ollama_reuses_installed_models(vault, mock_ollama):
    Wizard(vault, interactive=False, echo=lambda *_: None).run(provider="ollama", name="local", use="default")
    cfg = load_config(vault)
    assert cfg.llm.provider == "ollama" and cfg.llm.model == "qwen2.5:7b-instruct-q4_K_M"


def test_wizard_embeddings(vault, mock_ollama):
    Wizard(vault, interactive=False, echo=lambda *_: None).run(provider="ollama", model="bge-m3", embeddings=True)
    e = load_config(vault).embeddings
    assert (e.provider, e.model, e.enabled) == ("ollama", "bge-m3", True)


# ------------------------------------------------------------------ Ollama detection
def test_ollama_status_pull_and_check(mock_ollama):
    st = providers.ollama_status()
    assert st.running and st.version == "0.15.2" and st.has("bge-m3")
    msgs = []
    providers.ollama_pull("qwen3:4b", echo=msgs.append)
    assert any("50%" in m for m in msgs) and providers.ollama_status().has("qwen3:4b")
    missing = connect.check_profile("x", LLMProfile(provider="ollama", model="llama-absent"))
    assert not missing.ok and "pull" in missing.message
    assert connect.check_profile("x", LLMProfile(provider="ollama", model="deepseek-r1:14b"), live=True).ok


def test_ollama_not_running(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:1")
    st = providers.ollama_status()
    assert not st.running
    assert "not reachable" in connect.check_profile("x", LLMProfile(provider="ollama", model="m")).message


# ------------------------------------------------------------------ Claude Code (subscription)
FAKE_CLAUDE = r'''#!/usr/bin/env python3
import json, os, sys
args = sys.argv[1:]
if args[:2] == ["auth", "status"]:
    print(json.dumps({"loggedIn": True, "authMethod": "claude.ai"})); sys.exit(0)
system = open(args[args.index("--system-prompt-file") + 1]).read()
json.dump({"args": args, "cwd": os.getcwd(), "system": system, "prompt": sys.stdin.read(),
           "api_key": os.environ.get("ANTHROPIC_API_KEY")}, open(os.environ["FAKE_CLAUDE_LOG"], "w"))
print(json.dumps({"type": "result", "is_error": False, "result": "{\"ok\": true}"}))
'''


def test_claude_code_client_uses_subscription_without_tools(tmp_path, monkeypatch, vault):
    exe = tmp_path / "claude"
    exe.write_text(FAKE_CLAUDE)
    exe.chmod(0o755)
    log = tmp_path / "log.json"
    monkeypatch.setenv("FAKE_CLAUDE_LOG", str(log))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "should-not-leak")
    client = make_client(LLMProfile(provider="claude_code", model="opus", command=str(exe)))
    assert client.chat("SYSTEM PROMPT", "USER PROMPT", json_mode=True) == '{"ok": true}'
    call = json.loads(log.read_text())
    args = call["args"]
    assert args[:3] == ["-p", "--output-format", "json"] and args[args.index("--tools") + 1] == ""
    assert "--no-session-persistence" in args and args[args.index("--model") + 1] == "opus"
    assert call["system"] == "SYSTEM PROMPT" and call["prompt"] == "USER PROMPT"
    assert call["api_key"] is None                                   # subscription login, not the key
    assert str(vault) not in call["cwd"]                             # vault CLAUDE.md is not loaded
    assert providers.claude_code_status(str(exe)) == (True, "logged in (claude.ai)")


def test_claude_code_missing_binary():
    with pytest.raises(LLMError, match="install"):
        make_client(LLMProfile(provider="claude_code", command="/nonexistent/claude")).chat("s", "u")


# ------------------------------------------------------------------ LiteLLM
def test_litellm_client_with_fake_module(monkeypatch):
    calls = {}

    def completion(**kw):
        calls.update(kw)
        msg = types.SimpleNamespace(content="hi")
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    def embedding(**kw):
        return types.SimpleNamespace(data=[{"embedding": [1.0]} for _ in kw["input"]])

    monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(completion=completion, embedding=embedding))
    client = make_client(LLMProfile(provider="litellm", model="bedrock/some-model", json_mode=True))
    assert client.chat("s", "u", json_mode=True) == "hi"
    assert calls["model"] == "bedrock/some-model" and calls["response_format"] == {"type": "json_object"}
    assert client.embed(["a", "b"]) == [[1.0], [1.0]]


def test_litellm_missing_gives_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "litellm", None)  # import fails
    with pytest.raises(LLMError, match="uv tool install"):
        make_client(LLMProfile(provider="litellm", model="x")).chat("s", "u")


# ------------------------------------------------------------------ agent launcher
def test_agent_env_deepseek_hides_secret(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deep")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    env, shown, _ = agent_env("deepseek")
    assert env["ANTHROPIC_BASE_URL"] == "https://api.deepseek.com/anthropic"
    assert env["ANTHROPIC_AUTH_TOKEN"] == "sk-deep" and "ANTHROPIC_API_KEY" not in env
    assert env["ANTHROPIC_MODEL"] == "deepseek-flash[1m]" and env["CLAUDE_CODE_SUBAGENT_MODEL"] == "deepseek-flash"
    lines = export_lines(shown)
    assert '"$DEEPSEEK_API_KEY"' in lines and "sk-deep" not in lines
    assert agent_env("deepseek", model="deepseek-v4-pro")[0]["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "deepseek-v4-pro"


def test_agent_env_ollama_and_errors(mock_ollama, monkeypatch):
    env, _, notes = agent_env("ollama", default_ollama_model="deepseek-r1:14b")
    assert env["ANTHROPIC_BASE_URL"] == mock_ollama and env["ANTHROPIC_AUTH_TOKEN"] == "ollama"
    assert env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "deepseek-r1:14b" and any("context" in n for n in notes)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(ValueError, match="rbrain connect deepseek"):
        agent_env("deepseek")
    with pytest.raises(ValueError):
        agent_env("custom", base_url="https://x")
    assert agent_env("claude")[1] == {}


# ------------------------------------------------------------------ doctor and CLI
def test_doctor_marks_unused_profiles_optional(vault, monkeypatch, capsys):
    text = upsert_profile(config_path(vault).read_text(), "claude-api",
                          {"provider": "anthropic", "model": "claude-sonnet-5-5"})
    config_path(vault).write_text(text)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from research_brain.bootstrap import doctor
    results = doctor(load_config(vault))
    line = next(m for ok, m in results if "claude-api" in m)
    assert "optional" in line and all(ok for ok, _ in results)
    from research_brain.cli import main
    main(["--vault", str(vault), "connect", "--list"])
    assert "claude-api" in capsys.readouterr().out
