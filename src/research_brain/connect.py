"""`rbrain connect`: connect the vault to a model (interactively or with flags).

The wizard detects what is already available (a running Ollama and its models, a logged-in
Claude Code, API keys in the environment), asks for a key only when one is missing, stores
it outside the vault (``~/.config/research-brain/secrets.env``, mode 600), lists the models
the provider offers *live*, tests the connection with a one-line prompt, and writes a named
profile into ``.rbrain/config.toml`` — as the default, for selected skills, or as an option
for ``--profile``.
"""

from __future__ import annotations

import getpass
import importlib.util
import json
import os
import re
import shutil
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import providers
from .config import LLMProfile, config_path, load_config
from .llm import LLMError, make_client

SKILLS = ["ingest-source", "segment-idea", "synthesize-ideas", "explain-concept",
          "deductive-reasoning", "query-brain", "lint-brain"]
EMBEDDING_HINTS = ("embed", "bge", "nomic", "minilm", "e5-", "gte")
DEFAULT_PROFILE_NAMES = {"claude_code": "claude", "anthropic": "claude-api"}


# =========================================================================== TOML editing
def toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    return json.dumps(str(v), ensure_ascii=False)  # JSON string escapes are valid TOML


def _section_span(text: str, section: str) -> tuple[int, int] | None:
    """(start, end) character offsets of ``[section]`` up to the next table header."""
    m = re.search(rf"^\[{re.escape(section)}\][ \t]*(#.*)?$", text, re.M)
    if not m:
        return None
    nxt = re.search(r"^\[[^\]]+\][ \t]*(#.*)?$", text[m.end():], re.M)
    end = m.end() + nxt.start() if nxt else len(text)
    return m.start(), end


def upsert_profile(text: str, name: str, values: dict) -> str:
    """Create or replace ``[llm.profiles.<name>]``."""
    block = f"[llm.profiles.{name}]\n" + "".join(
        f"{k} = {toml_value(v)}\n" for k, v in values.items() if v not in (None, ""))
    span = _section_span(text, f"llm.profiles.{name}")
    if span:
        start, end = span
        return text[:start] + block + "\n" + text[end:].lstrip("\n")
    anchor = _section_span(text, "skill_profiles") or _section_span(text, "embeddings")
    if anchor:
        # Insert after the last real line before the anchor, so comment blocks that belong to
        # the next section (or commented-out example profiles) stay where they are.
        lines = text[:anchor[0]].split("\n")
        i = len(lines) - 1
        while i > 0 and (not lines[i - 1].strip() or lines[i - 1].lstrip().startswith("#")):
            i -= 1
        pos = len("\n".join(lines[:i])) + 1 if i > 0 else 0
        return text[:pos] + "\n" + block + text[pos:]
    return text.rstrip("\n") + "\n\n" + block


def set_key(text: str, section: str, key: str, value: str | None) -> str:
    """Set (or remove, with ``value=None``) ``key`` inside ``[section]``; a commented-out
    ``# key = ...`` line is reused so the file keeps its layout."""
    span = _section_span(text, section)
    line = f"{key} = {toml_value(value)}" if value is not None else None
    if span is None:
        if line is None:
            return text
        return text.rstrip("\n") + f"\n\n[{section}]\n{line}\n"
    start, end = span
    body = text[start:end]
    pattern = re.compile(rf"^[ \t]*#?[ \t]*{re.escape(key)}[ \t]*=.*$", re.M)
    m = pattern.search(body)
    if m:
        if line is None:
            if m.group(0).lstrip().startswith("#"):
                return text  # already absent
            new_body = body[:m.start()] + body[m.end() + 1:]
        else:
            new_body = body[:m.start()] + line + body[m.end():]
    else:
        if line is None:
            return text
        header_end = body.index("\n") + 1 if "\n" in body else len(body)
        new_body = body[:header_end] + line + "\n" + body[header_end:]
    return text[:start] + new_body + text[end:]


def write_config(vault: Path, new_text: str) -> None:
    """Validate, back up and atomically replace the vault's config.toml."""
    tomllib.loads(new_text)  # raises on syntax errors before anything is written
    path = config_path(vault)
    backup = path.with_suffix(".toml.bak")
    if path.exists():
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    tmp = path.with_suffix(".toml.tmp")
    tmp.write_text(new_text, encoding="utf-8")
    tmp.replace(path)
    try:
        load_config(vault)  # semantic check (unknown provider, missing profile...)
    except Exception:
        if backup.exists():
            backup.replace(path)
        raise


# =========================================================================== status
def provider_status(name: str) -> tuple[bool, str]:
    """Quick, offline-friendly status line for the provider menu."""
    spec = providers.get(name)
    if spec.api == "ollama":
        st = providers.ollama_status()
        if st.running:
            return True, f"running v{st.version} at {st.url}, {len(st.models)} models"
        if st.binary:
            return False, "installed but not running (systemctl start ollama, or ollama serve)"
        return False, "not found at " + st.url
    if name == "lmstudio":
        try:
            providers.http_get_json(f"{spec.base_url}/models", timeout=2)
            return True, "server running"
        except Exception:
            return False, "server not running"
    if spec.api == "claude_code":
        return providers.claude_code_status()
    if spec.api == "litellm":
        try:
            if importlib.util.find_spec("litellm") is not None:
                return True, "installed"
        except (ImportError, ValueError):
            pass
        return False, "optional extra not installed (bash install.sh --with-litellm)"
    if spec.api_key_env:
        ok = bool(os.environ.get(spec.api_key_env))
        return ok, "key found" if ok else f"needs {spec.api_key_env}"
    return True, ""


@dataclass
class CheckResult:
    ok: bool
    message: str


def check_profile(name: str, p: LLMProfile, live: bool = False) -> CheckResult:
    """Static checks (server up, model present, key or login available) and optional live call."""
    label = f"{name}: {p.provider}:{p.model or '?'}"
    api = p.spec.api
    if api == "fake":
        return CheckResult(True, f"{label} — test provider")
    if not p.model and api != "claude_code":
        return CheckResult(False, f"{label} — no model set (rbrain connect {p.provider})")
    if api == "ollama":
        st = providers.ollama_status(p.base_url)
        if not st.running:
            hint = ("start it: sudo systemctl start ollama (or ollama serve)" if st.binary
                    else "install it (https://ollama.com) or point OLLAMA_HOST to a running server")
            return CheckResult(False, f"{label} — Ollama not reachable at {st.url}; {hint}")
        if not st.has(p.model):
            return CheckResult(False, f"{label} — model missing: rbrain ollama --pull-missing "
                                      f"(or ollama pull {p.model})")
    elif api == "claude_code":
        ok, msg = providers.claude_code_status(p.command)
        if not ok:
            return CheckResult(False, f"{label} — {msg}")
    elif api == "litellm":
        ok, msg = provider_status("litellm")
        if not ok:
            return CheckResult(False, f"{label} — {msg}")
    elif p.api_key_env and not os.environ.get(p.api_key_env):
        return CheckResult(False, f"{label} — {p.api_key_env} not set: rbrain connect {p.provider}")
    elif p.provider in {"lmstudio", "openai_compat"}:
        try:
            providers.http_get_json(f"{p.base_url}/models", timeout=3)
        except Exception as exc:
            return CheckResult(False, f"{label} — server not reachable at {p.base_url} ({exc})")
    if not live:
        return CheckResult(True, f"{label} — ready")
    ok, msg = test_profile(p)
    return CheckResult(ok, f"{label} — {msg}")


def test_profile(p: LLMProfile) -> tuple[bool, str]:
    """Send a one-line prompt and report the latency."""
    t0 = time.time()
    try:
        reply = make_client(p).chat("You are a connection test.", "Reply with exactly the word: OK",
                                    tag="connect-test")
    except LLMError as exc:
        return False, f"call failed: {exc}"
    dt = time.time() - t0
    reply = reply.strip().replace("\n", " ")
    return bool(reply), f"answered '{reply[:30]}' in {dt:.1f} s"


def test_embeddings(p: LLMProfile) -> tuple[bool, str]:
    try:
        return True, f"embedding dimension {len(make_client(p).embed(['test'])[0])}"
    except LLMError as exc:
        return False, str(exc)


# =========================================================================== wizard
class Wizard:
    """Interactive (or flag-driven) connection setup. I/O functions are injectable for tests."""

    def __init__(self, vault: Path, ask: Callable[[str], str] = input,
                 ask_secret: Callable[[str], str] = getpass.getpass,
                 echo: Callable[[str], None] = print, interactive: bool = True):
        self.vault = vault
        self.ask_raw, self.ask_secret, self.echo = ask, ask_secret, echo
        self.interactive = interactive

    # ------------------------------------------------------------- small prompts
    def ask(self, question: str, default: str = "") -> str:
        if not self.interactive:
            return default
        answer = self.ask_raw(f"{question}{f' [{default}]' if default else ''}: ").strip()
        return answer or default

    def choose(self, question: str, options: list[str], default: int = 1) -> int:
        for i, opt in enumerate(options, 1):
            self.echo(f"  {i:>2}. {opt}")
        while True:
            raw = self.ask(question, str(default))
            if raw.isdigit() and 1 <= int(raw) <= len(options):
                return int(raw)
            self.echo("  Please type one of the numbers above.")

    def menu(self) -> list[str]:
        names = [n for n in providers.PROVIDERS if n != "fake"]
        families = {"local": "Local (private, free)", "claude": "Claude",
                    "cloud": "Cloud APIs (OpenAI-compatible)", "gateway": "Other"}
        lines, ordered = [], []
        for fam, title in families.items():
            lines.append(f"\n{title}")
            for n in names:
                spec = providers.get(n)
                if spec.family != fam:
                    continue
                ok, status = provider_status(n)
                ordered.append(n)
                lines.append(f"  {len(ordered):>2}. {n:<13} {spec.label}"
                             + (f"  [{'✔' if ok else '·'} {status}]" if status else ""))
        self.echo("\n".join(lines))
        return ordered

    # ------------------------------------------------------------- main flow
    def run(self, provider: str | None = None, name: str | None = None, model: str | None = None,
            base_url: str | None = None, key_env: str | None = None, use: str | None = None,
            skills: list[str] | None = None, test: bool = True, embeddings: bool = False) -> str:
        if provider is None:
            if not self.interactive:
                raise ValueError("Give a provider, e.g. `rbrain connect deepseek`.")
            ordered = self.menu()
            raw = self.ask("\nWhich connection? (number or name)")
            provider = ordered[int(raw) - 1] if raw.isdigit() and 0 < int(raw) <= len(ordered) else raw
        spec = providers.get(provider)

        if spec.name == "openai_compat":
            base_url = base_url or self.ask("Base URL of the server (e.g. http://localhost:8000/v1)",
                                            spec.base_url)
            key_env = key_env if key_env is not None else self.ask("Env variable with its API key "
                                                                    "(empty if none)", "")
        key_env = key_env if key_env is not None else spec.api_key_env
        if key_env:
            self._ensure_key(spec, key_env)
        model = model or self._pick_model(spec, base_url, key_env)
        if not model and spec.api != "claude_code":
            raise ValueError("No model chosen.")

        values = {"provider": spec.name, "model": model}
        if base_url and base_url != spec.base_url:
            values["base_url"] = base_url
        if key_env and key_env != spec.api_key_env:
            values["api_key_env"] = key_env
        profile = LLMProfile.from_dict(values)

        if test:
            self.echo(f"Testing {spec.name}:{model} …")
            ok, msg = test_embeddings(profile) if embeddings else test_profile(profile)
            self.echo(f"  {'✔' if ok else '✘'} {msg}")
            if not ok and self.ask("Save the profile anyway? (y/N)", "n").lower() != "y":
                raise LLMError(f"Connection test failed: {msg}")

        text = config_path(self.vault).read_text(encoding="utf-8")
        if embeddings:
            for k, v in [("enabled", True), ("provider", spec.name), ("model", model),
                         ("base_url", values.get("base_url")), ("api_key_env", values.get("api_key_env"))]:
                text = set_key(text, "embeddings", k, v)  # type: ignore[arg-type]
            write_config(self.vault, text)
            self.echo(f"Embeddings now use {spec.name}:{model}. Rebuild the cache: rbrain index --embed")
            return "embeddings"

        name = name or (self.ask("Profile name", DEFAULT_PROFILE_NAMES.get(spec.name, spec.name))
                        if self.interactive else DEFAULT_PROFILE_NAMES.get(spec.name, spec.name))
        name = re.sub(r"[^A-Za-z0-9_-]", "-", name)
        text = upsert_profile(text, name, values)

        if use is None:
            use = ["default", "skills", "optional"][self.choose(
                "How should this profile be used?",
                ["as the default for every skill",
                 "only for some skills (e.g. synthesis and deduction)",
                 f"only on demand (rbrain --profile {name} ...)"], default=3) - 1] if self.interactive else "optional"
        if use == "default":
            text = set_key(text, "llm", "default_profile", name)
        elif use == "skills":
            if skills is None:
                idx = self.ask("Skills (numbers separated by commas): " + ", ".join(
                    f"{i}={s}" for i, s in enumerate(SKILLS, 1)), "3,5")
                skills = [SKILLS[int(i) - 1] for i in re.findall(r"\d+", idx) if 0 < int(i) <= len(SKILLS)]
            for s in skills:
                text = set_key(text, "skill_profiles", s, name)
        write_config(self.vault, text)

        self.echo(f"\nSaved profile [{name}] → {spec.name}:{model}")
        if use == "default":
            self.echo("It is now the default model for every skill.")
        elif use == "skills":
            self.echo(f"Used by: {', '.join(skills or [])}")
        self.echo(f"Use it for one command:  rbrain --profile {name} explain \"inverse problem\"")
        return name

    # ------------------------------------------------------------- helpers
    def _ensure_key(self, spec: providers.Provider, key_env: str) -> None:
        if os.environ.get(key_env):
            self.echo(f"  ✔ {key_env} found ({providers.mask(os.environ[key_env])})")
            return
        if not self.interactive:
            raise ValueError(f"{key_env} is not set. Export it or run `rbrain connect {spec.name}` interactively.")
        if spec.key_url:
            self.echo(f"  Create a key at {spec.key_url}")
        key = self.ask_secret(f"  Paste your {key_env} (hidden): ").strip()
        if not key:
            raise ValueError("No key given.")
        path = providers.save_secret(key_env, key)
        self.echo(f"  ✔ saved to {path} (permissions 600, outside the vault)")

    def _pick_model(self, spec: providers.Provider, base_url: str | None, key_env: str | None) -> str:
        if spec.api == "ollama":
            st = providers.ollama_status(base_url or "")
            if not st.running:
                self.echo(f"  ✘ Ollama is not reachable at {st.url}. "
                          + ("Start it with: sudo systemctl start ollama (or ollama serve)" if st.binary
                             else "Install it from https://ollama.com or set OLLAMA_HOST."))
                return self.ask("Model name to use anyway", spec.default_model)
            chat_models = [m for m in st.models if not any(h in m.lower() for h in EMBEDDING_HINTS)]
            if not chat_models:
                self.echo("  No chat models installed yet.")
                wanted = self.ask("Model to pull", spec.default_model)
                if self.interactive and self.ask(f"Pull {wanted} now? (Y/n)", "y").lower() != "n":
                    providers.ollama_pull(wanted, st.url, self.echo)
                return wanted
            default = spec.default_model if spec.default_model in chat_models else chat_models[0]
            if not self.interactive:
                return default
            self.echo("  Installed models:")
            i = self.choose("Model", chat_models, chat_models.index(default) + 1)
            return chat_models[i - 1]
        if spec.api == "litellm":
            return self.ask("LiteLLM model id (e.g. bedrock/anthropic.claude-..., vertex_ai/gemini-..., "
                            "azure/<deployment>)")
        models = providers.list_models(spec.name, base_url or "", key_env or "")
        if spec.default_model and spec.default_model not in models:
            models = [spec.default_model, *models]
        if not models:
            return self.ask("Model id", spec.default_model)
        if not self.interactive:
            return spec.default_model or models[0]
        shown = models[:30]
        self.echo(f"  Models offered by {spec.name}" + (f" (first 30 of {len(models)})" if len(models) > 30 else "") + ":")
        default = shown.index(spec.default_model) + 1 if spec.default_model in shown else 1
        i = self.choose("Model (or type 0 to enter another id)", shown + ["(other id)"], default)
        return shown[i - 1] if i <= len(shown) else self.ask("Model id")


def list_connections(vault: Path, echo: Callable[[str], None] = print, live: bool = False) -> None:
    cfg = load_config(vault)
    echo("Configured profiles:")
    for name, p in cfg.all_profiles().items():
        r = check_profile(name, p, live=live)
        echo(f"  {'✔' if r.ok else '✘'} {r.message}")
    if cfg.skill_profiles:
        echo("Skill → profile: " + ", ".join(f"{k}→{v}" for k, v in cfg.skill_profiles.items()))
    e = cfg.embeddings
    echo(f"Embeddings: {e.provider}:{e.model}" + ("" if e.enabled else " (disabled)"))
    if shutil.which("claude"):
        echo("Agent mode: rbrain agent [--backend claude|deepseek|ollama]")
