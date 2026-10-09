"""`rbrain agent`: open Claude Code in the vault, backed by Claude or by another model.

Claude Code speaks the Anthropic Messages API, and several providers now expose an
Anthropic-compatible endpoint. Pointing ``ANTHROPIC_BASE_URL`` at one of them runs the
*agent mode* of the second brain (Claude Code reading CLAUDE.md and using the skills) on a
different model:

* ``claude``   – your Claude login (Pro/Max subscription or Console key). Default.
* ``deepseek`` – DeepSeek's Anthropic-compatible endpoint (DEEPSEEK_API_KEY).
* ``ollama``   – a local model through Ollama's Anthropic-compatible API.
* ``custom``   – any other Anthropic-compatible endpoint (``--base-url``, ``--token-env``).

Values follow the providers' own Claude Code guides (DeepSeek: api-docs.deepseek.com,
Ollama: docs.ollama.com/api/anthropic-compatibility).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from .providers import ollama_status, ollama_url

CLAUDE_VARS = ("ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_MODEL", "ANTHROPIC_DEFAULT_OPUS_MODEL",
               "ANTHROPIC_DEFAULT_SONNET_MODEL", "ANTHROPIC_DEFAULT_HAIKU_MODEL", "CLAUDE_CODE_SUBAGENT_MODEL")
BACKENDS = ("claude", "deepseek", "ollama", "custom")


def agent_env(backend: str, model: str | None = None, base_url: str | None = None,
              token_env: str | None = None, default_ollama_model: str = "") -> tuple[dict, dict, list[str]]:
    """Return (full environment, the variables that were set, notes for the user).

    In the returned "set" dict, secrets are shown as ``$VARIABLE`` references, never as values.
    """
    env = {k: v for k, v in os.environ.items() if k not in CLAUDE_VARS}
    shown: dict[str, str] = {}
    notes: list[str] = []

    def put(key: str, value: str, display: str | None = None) -> None:
        env[key] = value
        shown[key] = display if display is not None else value

    if backend == "claude":
        if model:
            put("ANTHROPIC_MODEL", model)
        notes.append("Using your Claude login (run `claude` once to sign in, or set ANTHROPIC_API_KEY).")
    elif backend == "deepseek":
        key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not key:
            raise ValueError("DEEPSEEK_API_KEY is not set: run `rbrain connect deepseek` first.")
        env.pop("ANTHROPIC_API_KEY", None)
        main, fast = model or "deepseek-flash[1m]", "deepseek-flash"
        put("ANTHROPIC_BASE_URL", "https://api.deepseek.com/anthropic")
        put("ANTHROPIC_AUTH_TOKEN", key, "$DEEPSEEK_API_KEY")
        for var in ("ANTHROPIC_MODEL", "ANTHROPIC_DEFAULT_OPUS_MODEL", "ANTHROPIC_DEFAULT_SONNET_MODEL"):
            put(var, main)
        put("ANTHROPIC_DEFAULT_HAIKU_MODEL", fast)
        put("CLAUDE_CODE_SUBAGENT_MODEL", fast)
        notes.append("Billed to your DeepSeek account. Use --model deepseek-v4-pro for the larger model.")
    elif backend == "ollama":
        model = model or default_ollama_model
        if not model:
            raise ValueError("Give a model: rbrain agent --backend ollama --model <installed model>")
        st = ollama_status()
        if not st.running:
            raise ValueError(f"Ollama is not reachable at {st.url}.")
        if not st.has(model):
            notes.append(f"'{model}' is not installed in Ollama: rbrain ollama --pull {model}")
        env.pop("ANTHROPIC_API_KEY", None)
        put("ANTHROPIC_BASE_URL", ollama_url())
        put("ANTHROPIC_AUTH_TOKEN", "ollama")  # required by Claude Code, ignored by Ollama
        for var in CLAUDE_VARS[2:]:
            put(var, model)
        notes += [
            "Claude Code sends long prompts and relies on tool calling: prefer models of 14B+ with good "
            "tool use, and give Ollama a large context (e.g. OLLAMA_CONTEXT_LENGTH=32768 for the server).",
            "If Claude Code reports a 404 on /v1/messages, update Ollama (the Anthropic API is recent).",
        ]
    elif backend == "custom":
        if not (base_url and token_env and model):
            raise ValueError("custom backend needs --base-url, --token-env and --model")
        token = os.environ.get(token_env, "")
        if not token:
            raise ValueError(f"{token_env} is not set.")
        env.pop("ANTHROPIC_API_KEY", None)
        put("ANTHROPIC_BASE_URL", base_url)
        put("ANTHROPIC_AUTH_TOKEN", token, f"${token_env}")
        for var in CLAUDE_VARS[2:]:
            put(var, model)
    else:
        raise ValueError(f"Unknown backend '{backend}'. Use one of: {', '.join(BACKENDS)}")
    return env, shown, notes


def export_lines(shown: dict[str, str]) -> str:
    """Shell lines to reproduce the setup by hand (secrets stay as $VARIABLE references)."""
    out = []
    for k, v in shown.items():
        out.append(f'export {k}="{v}"' if v.startswith("$") else f"export {k}='{v}'")
    return "\n".join(out)


def launch(vault: Path, env: dict, extra_args: list[str]) -> None:
    """Replace the current process with Claude Code running inside the vault."""
    exe = shutil.which("claude")
    if not exe:
        raise FileNotFoundError("Claude Code is not installed: curl -fsSL https://claude.ai/install.sh | bash")
    os.chdir(vault)
    os.execvpe(exe, [exe, *extra_args], env)
