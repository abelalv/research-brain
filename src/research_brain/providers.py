"""Registry of the ways research-brain can connect to a model, plus detection helpers.

Four families of connection are supported:

* **local servers**      – Ollama, LM Studio (and any OpenAI-compatible local server);
* **Claude**             – through *Claude Code* with your Pro/Max subscription (no API key),
                           or through the Claude API with a Console key;
* **cloud APIs**         – DeepSeek, OpenRouter, Gemini, Groq, Mistral, OpenAI, Together
                           (all OpenAI-compatible, so one client covers them);
* **gateway**            – LiteLLM (optional extra) for 100+ other providers
                           (Bedrock, Vertex AI, Azure OpenAI, Cohere, ...).

Model names change often, so the registry only suggests a default where the provider's
documentation is stable; ``rbrain connect`` lists the models the provider offers *live*.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

SECRETS_FILE = Path.home() / ".config" / "research-brain" / "secrets.env"


@dataclass(frozen=True)
class Provider:
    name: str
    label: str
    family: str                 # local | claude | cloud | gateway | test
    api: str                    # ollama | openai | anthropic | claude_code | litellm | fake
    base_url: str = ""
    api_key_env: str = ""
    default_model: str = ""
    json_mode: bool = False     # send response_format=json_object by default
    embeddings: bool = False    # offers an embeddings endpoint
    key_url: str = ""           # where to create an API key
    notes: str = ""
    suggested_models: tuple[str, ...] = field(default_factory=tuple)


PROVIDERS: dict[str, Provider] = {p.name: p for p in [
    Provider("ollama", "Ollama (local models)", "local", "ollama", "http://localhost:11434",
             default_model="qwen2.5:7b-instruct-q4_K_M", embeddings=True,
             notes="Private and free. Uses the models you have pulled."),
    Provider("lmstudio", "LM Studio (local server)", "local", "openai", "http://localhost:1234/v1",
             embeddings=True, notes="Start the local server in LM Studio (Developer tab)."),
    Provider("claude_code", "Claude via Claude Code (your Pro/Max subscription, no API key)", "claude",
             "claude_code", default_model="sonnet",
             suggested_models=("sonnet", "opus", "haiku"),
             notes="Runs `claude -p` headless with tools disabled. Counts against your plan's usage limits."),
    Provider("anthropic", "Claude API (Console API key)", "claude", "anthropic",
             "https://api.anthropic.com", "ANTHROPIC_API_KEY", default_model="claude-sonnet-5-5",
             key_url="https://platform.claude.com", notes="Pay-per-use API billing."),
    Provider("deepseek", "DeepSeek API", "cloud", "openai", "https://api.deepseek.com",
             "DEEPSEEK_API_KEY", default_model="deepseek-flash", json_mode=True,
             suggested_models=("deepseek-flash", "deepseek-v4-pro"),
             key_url="https://platform.deepseek.com/api_keys", notes="Low cost, strong reasoning, JSON mode."),
    Provider("openrouter", "OpenRouter (hundreds of models with one key)", "cloud", "openai",
             "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", key_url="https://openrouter.ai/keys",
             notes="Model ids look like 'provider/model'."),
    Provider("gemini", "Google Gemini (OpenAI-compatible endpoint)", "cloud", "openai",
             "https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY",
             embeddings=True, key_url="https://aistudio.google.com/apikey"),
    Provider("groq", "Groq (very fast open models)", "cloud", "openai", "https://api.groq.com/openai/v1",
             "GROQ_API_KEY", key_url="https://console.groq.com/keys"),
    Provider("mistral", "Mistral AI", "cloud", "openai", "https://api.mistral.ai/v1", "MISTRAL_API_KEY",
             json_mode=True, embeddings=True, key_url="https://console.mistral.ai/api-keys"),
    Provider("openai", "OpenAI", "cloud", "openai", "https://api.openai.com/v1", "OPENAI_API_KEY",
             json_mode=True, embeddings=True, key_url="https://platform.openai.com/api-keys"),
    Provider("together", "Together AI (open models)", "cloud", "openai", "https://api.together.xyz/v1",
             "TOGETHER_API_KEY", embeddings=True, key_url="https://api.together.ai/settings/api-keys"),
    Provider("openai_compat", "Any OpenAI-compatible server (vLLM, llama.cpp, ...)", "gateway", "openai",
             "http://localhost:8000/v1", "", notes="Set base_url (and api_key_env if needed)."),
    Provider("litellm", "LiteLLM gateway (Bedrock, Vertex AI, Azure, Cohere, 100+ more)", "gateway",
             "litellm", notes="Optional extra. Model ids like 'bedrock/...', 'vertex_ai/...', 'azure/...'."),
    Provider("fake", "Canned responses (tests and demo)", "test", "fake"),
]}


def get(name: str) -> Provider:
    if name not in PROVIDERS:
        raise ValueError(f"Unknown provider '{name}'. Use one of: {', '.join(PROVIDERS)}")
    return PROVIDERS[name]


# --------------------------------------------------------------------------- secrets
def load_secrets(path: Path | None = None) -> dict[str, str]:
    """Load KEY=VALUE lines into os.environ (existing variables win). Returns what was read."""
    path = path or SECRETS_FILE
    found: dict[str, str] = {}
    if not path.exists():
        return found
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip().removeprefix("export ").strip(), value.strip().strip('"').strip("'")
        found[key] = value
        os.environ.setdefault(key, value)
    return found


def save_secret(key: str, value: str, path: Path | None = None) -> Path:
    """Store an API key outside the vault (the vault may be synced or committed), mode 600."""
    path = path or SECRETS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    if path.exists():
        lines = [ln for ln in path.read_text(encoding="utf-8").splitlines()
                 if not ln.strip().startswith(f"{key}=") and not ln.strip().startswith(f"export {key}=")]
    else:
        lines = ["# research-brain API keys (chmod 600). Loaded automatically by rbrain."]
    lines.append(f"{key}={value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.chmod(path, 0o600)
    os.environ[key] = value
    return path


def mask(secret: str) -> str:
    return secret[:4] + "…" + secret[-4:] if len(secret) > 10 else "…"


# --------------------------------------------------------------------------- HTTP helpers
def http_get_json(url: str, headers: dict | None = None, timeout: int = 10) -> dict:
    req = urllib.request.Request(url)
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def ollama_url(base_url: str = "") -> str:
    """Ollama base URL, honouring OLLAMA_HOST (e.g. a server on the Windows side of WSL)."""
    if base_url:
        return base_url.rstrip("/")
    host = os.environ.get("OLLAMA_HOST", "").strip()
    if host:
        if not host.startswith("http"):
            host = "http://" + host
        if host.count(":") == 1:  # no port given
            host += ":11434"
        return host.rstrip("/")
    return "http://localhost:11434"


@dataclass
class OllamaStatus:
    url: str
    running: bool
    version: str = ""
    models: list[str] = field(default_factory=list)
    binary: str | None = None
    error: str = ""

    def has(self, model: str) -> bool:
        return model in self.models or f"{model}:latest" in self.models


def ollama_status(base_url: str = "") -> OllamaStatus:
    """Detect Ollama by its API first (works for a server on another host or on Windows from WSL),
    then by its binary."""
    url = ollama_url(base_url)
    status = OllamaStatus(url=url, running=False, binary=shutil.which("ollama"))
    try:
        status.version = http_get_json(f"{url}/api/version", timeout=3).get("version", "")
        status.running = True
        status.models = [m["name"] for m in http_get_json(f"{url}/api/tags", timeout=5).get("models", [])]
    except (urllib.error.URLError, OSError, ValueError) as exc:
        status.error = str(exc)
    return status


def ollama_pull(model: str, base_url: str = "", echo: Callable[[str], None] = print) -> None:
    """Pull a model through the API (works even when the Ollama CLI is not on this machine)."""
    url = ollama_url(base_url)
    req = urllib.request.Request(f"{url}/api/pull", data=json.dumps({"model": model, "stream": True}).encode(),
                                 method="POST")
    req.add_header("Content-Type", "application/json")
    last = ""
    with urllib.request.urlopen(req, timeout=24 * 3600) as resp:
        for raw in resp:
            if not raw.strip():
                continue
            event = json.loads(raw)
            if "error" in event:
                raise RuntimeError(event["error"])
            msg = event.get("status", "")
            if event.get("total") and event.get("completed"):
                pct = int(100 * event["completed"] / event["total"])
                msg = f"{msg} {pct // 10 * 10}%"
            if msg and msg != last:
                echo(f"    {model}: {msg}")
                last = msg


# --------------------------------------------------------------------------- Claude Code
def claude_code_status(command: str = "") -> tuple[bool, str]:
    """(usable, message) for the Claude Code CLI, using `claude auth status` when available."""
    exe = shutil.which(command or "claude")
    if not exe:
        return False, "Claude Code is not installed (curl -fsSL https://claude.ai/install.sh | bash)"
    try:
        out = subprocess.run([exe, "auth", "status"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"cannot run {exe}: {exc}"
    if out.returncode == 0:
        try:
            method = json.loads(out.stdout).get("authMethod", "logged in")
        except json.JSONDecodeError:
            method = "logged in"
        return True, f"logged in ({method})"
    if "unknown" in (out.stderr + out.stdout).lower() or "did you mean" in (out.stderr + out.stdout).lower():
        return True, "installed (older version: login status not checked)"
    return False, "not logged in: run `claude` once (or `claude auth login`)"


# --------------------------------------------------------------------------- listing models
def list_models(provider: str, base_url: str = "", api_key_env: str = "") -> list[str]:
    """Models the provider offers right now (empty list if it cannot be queried)."""
    p = get(provider)
    base = (base_url or p.base_url).rstrip("/")
    key = os.environ.get(api_key_env or p.api_key_env, "") if (api_key_env or p.api_key_env) else ""
    try:
        if p.api == "ollama":
            return ollama_status(base).models
        if p.api == "claude_code":
            return list(p.suggested_models)
        if p.api == "anthropic":
            data = http_get_json(f"{base}/v1/models", {"x-api-key": key, "anthropic-version": "2023-06-01"})
            return [m["id"] for m in data.get("data", [])]
        if p.api == "openai":
            data = http_get_json(f"{base}/models", {"Authorization": f"Bearer {key}"} if key else {})
            return sorted(m.get("id", "") for m in data.get("data", []) if m.get("id"))
    except (urllib.error.URLError, OSError, ValueError, KeyError):
        return list(p.suggested_models)
    return list(p.suggested_models)
