"""Provider-agnostic LLM client.

One small interface (``chat`` and ``embed``) over six back-ends (see ``providers.py``):

* ``OllamaClient``       – local models through the Ollama REST API
* ``OpenAICompatClient`` – LM Studio, DeepSeek, OpenRouter, Gemini, Groq, Mistral, OpenAI,
                           Together, vLLM, llama.cpp server...
* ``AnthropicClient``    – Claude through the Messages API (API key)
* ``ClaudeCodeClient``   – Claude through the Claude Code CLI (your subscription, no API key)
* ``LiteLLMClient``      – optional gateway to 100+ providers (Bedrock, Vertex AI, Azure...)
* ``FakeClient``         – canned responses from disk, for tests and offline demos

Only the Python standard library is used for HTTP, so there is nothing extra to install.
The module also contains the JSON-extraction helpers that make small local models usable
for structured output (they often wrap JSON in prose, add trailing commas, or write LaTeX
with single backslashes such as ``\\frac`` which JSON silently turns into a form feed).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path

from .config import LLMProfile


class LLMError(RuntimeError):
    """Raised when a provider call fails or returns unusable output."""


# --------------------------------------------------------------------------- HTTP helper
def _post_json(url: str, payload: dict, headers: dict | None = None, timeout: int = 900) -> dict:
    """POST a JSON payload and return the decoded JSON response (stdlib only)."""
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise LLMError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise LLMError(f"Cannot reach {url}: {exc.reason}. Is the server running?") from exc


def _get_json(url: str, timeout: int = 10) -> dict:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError) as exc:
        raise LLMError(f"Cannot reach {url}: {exc}") from exc


# --------------------------------------------------------------------------- clients
class LLMClient(ABC):
    """Common interface. ``tag`` names the calling skill (used by logs and the fake client)."""

    def __init__(self, profile: LLMProfile):
        self.profile = profile

    @abstractmethod
    def chat(self, system: str, user: str, *, json_mode: bool = False, tag: str = "") -> str:
        """Return the assistant's text reply."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise LLMError(f"Provider '{self.profile.provider}' does not provide embeddings.")

    def api_key(self) -> str:
        env = self.profile.api_key_env
        if not env:
            return ""
        key = os.environ.get(env, "")
        if not key:
            raise LLMError(f"Environment variable {env} is not set (needed for {self.profile.provider}).")
        return key


class OllamaClient(LLMClient):
    def chat(self, system: str, user: str, *, json_mode: bool = False, tag: str = "") -> str:
        p = self.profile
        payload: dict = {
            "model": p.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "options": {"temperature": p.temperature, "num_ctx": p.num_ctx, "num_predict": p.max_tokens},
        }
        if json_mode and p.json_mode is not False:
            payload["format"] = "json"  # constrains decoding to valid JSON
        if p.think is not None:
            payload["think"] = p.think
        data = _post_json(f"{p.base_url}/api/chat", payload, timeout=p.timeout)
        return data.get("message", {}).get("content", "")

    def embed(self, texts: list[str]) -> list[list[float]]:
        p = self.profile
        data = _post_json(f"{p.base_url}/api/embed", {"model": p.model, "input": texts}, timeout=p.timeout)
        return data["embeddings"]

    def list_models(self) -> list[str]:
        data = _get_json(f"{self.profile.base_url}/api/tags")
        return [m["name"] for m in data.get("models", [])]


class OpenAICompatClient(LLMClient):
    """LM Studio, DeepSeek, vLLM, llama.cpp server, OpenRouter, OpenAI..."""

    def _headers(self) -> dict:
        key = self.api_key()
        return {"Authorization": f"Bearer {key}"} if key else {}

    def chat(self, system: str, user: str, *, json_mode: bool = False, tag: str = "") -> str:
        p = self.profile
        payload: dict = {
            "model": p.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": p.temperature,
            "max_tokens": p.max_tokens,
            "stream": False,
        }
        # json_object mode is supported by DeepSeek/OpenAI but not by every local server,
        # so it is only sent when the provider is known to accept it or the user enables it.
        use_json = p.json_mode if p.json_mode is not None else p.provider in {"deepseek"}
        if json_mode and use_json:
            payload["response_format"] = {"type": "json_object"}
        data = _post_json(f"{p.base_url}/chat/completions", payload, self._headers(), timeout=p.timeout)
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError) as exc:
            raise LLMError(f"Unexpected response: {str(data)[:300]}") from exc

    def embed(self, texts: list[str]) -> list[list[float]]:
        p = self.profile
        data = _post_json(f"{p.base_url}/embeddings", {"model": p.model, "input": texts},
                          self._headers(), timeout=p.timeout)
        return [row["embedding"] for row in data["data"]]


class AnthropicClient(LLMClient):
    API_VERSION = "2023-06-01"

    def chat(self, system: str, user: str, *, json_mode: bool = False, tag: str = "") -> str:
        p = self.profile
        payload = {
            "model": p.model,
            "max_tokens": p.max_tokens,
            "temperature": p.temperature,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        headers = {"x-api-key": self.api_key(), "anthropic-version": self.API_VERSION}
        data = _post_json(f"{p.base_url}/v1/messages", payload, headers, timeout=p.timeout)
        blocks = data.get("content", [])
        return "".join(b.get("text", "") for b in blocks if b.get("type") == "text")


class FakeClient(LLMClient):
    """Returns canned replies from ``$RBRAIN_FAKE_DIR/<tag>.json`` (or ``.md``).

    Used by the test-suite and by ``examples/demo.sh`` so the whole pipeline can be
    exercised without a model. Embeddings are deterministic hashed bag-of-words vectors.
    """

    def __init__(self, profile: LLMProfile, responses: dict[str, str] | None = None):
        super().__init__(profile)
        self.responses = responses or {}
        self.calls: list[tuple[str, str, str]] = []

    def chat(self, system: str, user: str, *, json_mode: bool = False, tag: str = "") -> str:
        self.calls.append((tag, system, user))
        if tag in self.responses:
            return self.responses[tag]
        folder = Path(os.environ.get("RBRAIN_FAKE_DIR", ""))
        for ext in (".json", ".md"):
            path = folder / f"{tag}{ext}"
            if path.exists():
                return path.read_text(encoding="utf-8")
        raise LLMError(f"FakeClient has no response for tag '{tag}' (looked in '{folder}').")

    def embed(self, texts: list[str]) -> list[list[float]]:
        dim = 256
        vectors = []
        for text in texts:
            vec = [0.0] * dim
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                h = int(hashlib.md5(word.encode()).hexdigest(), 16)
                vec[h % dim] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])
        return vectors


class ClaudeCodeClient(LLMClient):
    """Claude through the Claude Code CLI, authenticated with your Claude subscription.

    Runs ``claude -p`` headless from an empty temporary directory (so the vault's CLAUDE.md,
    hooks and MCP servers are not loaded), replaces the system prompt with the skill prompt,
    disables every tool, and reads the ``result`` field of the JSON output. No API key is
    needed; usage counts against your Pro/Max plan limits. ``ANTHROPIC_API_KEY`` and custom
    base URLs are removed from the child environment so the subscription login is used
    (set ``api_key_env`` in the profile to use an API key instead).
    """

    STRIP_ENV = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL", "ANTHROPIC_MODEL")

    def chat(self, system: str, user: str, *, json_mode: bool = False, tag: str = "") -> str:
        p = self.profile
        exe = shutil.which(p.command or "claude")
        if not exe:
            raise LLMError("Claude Code is not installed: curl -fsSL https://claude.ai/install.sh | bash")
        env = {k: v for k, v in os.environ.items() if k not in self.STRIP_ENV}
        if p.api_key_env:
            env["ANTHROPIC_API_KEY"] = self.api_key()
        with tempfile.TemporaryDirectory(prefix="rbrain-claude-") as tmp:
            prompt_file = Path(tmp) / "system-prompt.md"
            prompt_file.write_text(system, encoding="utf-8")
            cmd = [exe, "-p", "--output-format", "json", "--system-prompt-file", str(prompt_file),
                   "--tools", "", "--disallowedTools", "mcp__*", "--no-session-persistence"]
            if p.model:
                cmd += ["--model", p.model]
            try:
                proc = subprocess.run(cmd, input=user, capture_output=True, text=True, cwd=tmp,
                                      env=env, timeout=p.timeout)
            except subprocess.TimeoutExpired as exc:
                raise LLMError(f"Claude Code did not answer within {p.timeout} s") from exc
        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise LLMError(f"Claude Code failed (exit {proc.returncode}): "
                           f"{(proc.stderr or proc.stdout)[:400]}") from exc
        if proc.returncode != 0 or data.get("is_error"):
            raise LLMError(f"Claude Code error: {str(data.get('result') or proc.stderr)[:400]}")
        return str(data.get("result", ""))


class LiteLLMClient(LLMClient):
    """Any provider supported by LiteLLM (Bedrock, Vertex AI, Azure OpenAI, Cohere, ...).

    Optional dependency: ``uv tool install --editable . --with 'litellm!=1.82.7,!=1.82.8'``.
    LiteLLM reads the usual provider variables itself (AWS_*, AZURE_*, VERTEXAI_*, ...).
    """

    def _litellm(self):
        try:
            import litellm  # type: ignore[import-not-found]
        except ImportError as exc:
            raise LLMError("LiteLLM is not installed. Reinstall with the extra: "
                           "uv tool install --editable . --with 'litellm!=1.82.7,!=1.82.8'") from exc
        return litellm

    def _kwargs(self) -> dict:
        p = self.profile
        kw: dict = {"model": p.model, "timeout": p.timeout}
        if p.base_url:
            kw["api_base"] = p.base_url
        if p.api_key_env:
            kw["api_key"] = self.api_key()
        return kw

    def chat(self, system: str, user: str, *, json_mode: bool = False, tag: str = "") -> str:
        p = self.profile
        kw = self._kwargs() | {
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": p.temperature, "max_tokens": p.max_tokens,
        }
        if json_mode and p.json_mode:
            kw["response_format"] = {"type": "json_object"}
        try:
            resp = self._litellm().completion(**kw)
        except LLMError:
            raise
        except Exception as exc:  # LiteLLM raises many provider-specific exception types
            raise LLMError(f"LiteLLM ({p.model}): {exc}") from exc
        return resp.choices[0].message.content or ""

    def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            resp = self._litellm().embedding(input=texts, **self._kwargs())
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"LiteLLM embeddings ({self.profile.model}): {exc}") from exc
        return [row["embedding"] for row in resp.data]


def make_client(profile: LLMProfile) -> LLMClient:
    """Factory: build the right client for a profile (by the provider's API family)."""
    api = profile.spec.api
    clients = {"ollama": OllamaClient, "openai": OpenAICompatClient, "anthropic": AnthropicClient,
               "claude_code": ClaudeCodeClient, "litellm": LiteLLMClient, "fake": FakeClient}
    if api not in clients:
        raise LLMError(f"Unknown provider {profile.provider}")
    return clients[api](profile)


# --------------------------------------------------------------------------- output cleaning
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_think(text: str) -> str:
    """Remove <think>...</think> blocks emitted by reasoning models (deepseek-r1, qwen3)."""
    text = _THINK_RE.sub("", text)
    if "</think>" in text:  # opening tag lost (some servers strip it)
        text = text.split("</think>", 1)[1]
    return text.strip()


# LaTeX commands whose first letter collides with a JSON escape (\b \f \n \r \t \u).
# A model writing "$\frac{a}{b}$" inside a JSON string produces a *valid* escape \f
# (form feed) and silently corrupts the math, so these are doubled before parsing.
_LATEX_WORDS = {
    "beta", "bar", "bf", "big", "bigl", "bigr", "bigg", "boldsymbol", "binom", "bmatrix",
    "begin", "bullet", "backslash", "bot", "boxed", "breve", "bigcup", "bigcap",
    "frac", "forall", "flat", "footnotesize",
    "nabla", "nu", "neq", "ne", "not", "notin", "newline", "nonumber", "norm", "neg",
    "nolimits", "nearrow", "nexists", "ni",
    "rho", "right", "rangle", "rightarrow", "rm", "rceil", "rfloor", "rvert", "rVert",
    "theta", "tau", "text", "textbf", "textit", "textrm", "times", "tilde", "to", "top",
    "triangle", "tfrac", "tan", "tanh", "therefore", "texttt",
    "underline", "uparrow", "upsilon", "underbrace", "cup",
}


def repair_latex_escapes(raw: str) -> str:
    """Double single backslashes that belong to LaTeX rather than to JSON escapes."""
    out: list[str] = []
    i, n = 0, len(raw)
    while i < n:
        ch = raw[i]
        if ch != "\\" or i + 1 >= n:
            out.append(ch)
            i += 1
            continue
        nxt = raw[i + 1]
        if nxt == "\\":                       # already escaped backslash
            out.append("\\\\")
            i += 2
            continue
        if nxt in '"/':                        # legitimate \" or \/
            out.append(raw[i:i + 2])
            i += 2
            continue
        word_match = re.match(r"[A-Za-z]+", raw[i + 1:])
        word = word_match.group(0) if word_match else ""
        if nxt in "bfnrtu" and word in _LATEX_WORDS:
            out.append("\\\\")                 # LaTeX command, not an escape
            i += 1
            continue
        if nxt == "u" and re.match(r"u[0-9a-fA-F]{4}", raw[i + 1:]):
            out.append(raw[i:i + 6])           # genuine \uXXXX
            i += 6
            continue
        if nxt in "bfnrt":                     # genuine escape (\n, \t, ...)
            out.append(raw[i:i + 2])
            i += 2
            continue
        out.append("\\\\")                     # invalid JSON escape, e.g. \alpha, \partial
        i += 1
    return "".join(out)


def _balanced_object(text: str) -> str | None:
    """Return the first balanced {...} block in ``text`` (string-aware)."""
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for j in range(start, len(text)):
            c = text[j]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start:j + 1]
        start = text.find("{", start + 1)
    return None


def extract_json(text: str) -> dict:
    """Extract and parse the JSON object contained in an LLM reply."""
    text = strip_think(text)
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else _balanced_object(text)
    if candidate is None:
        raise ValueError("No JSON object found in the model output.")
    candidate = repair_latex_escapes(candidate)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # Common small-model slips: trailing commas and smart quotes.
        cleaned = re.sub(r",\s*([}\]])", r"\1", candidate)
        cleaned = cleaned.replace("“", '"').replace("”", '"')
        return json.loads(cleaned)


def chat_json(client: LLMClient, system: str, user: str, *, tag: str, retries: int = 1) -> dict:
    """Ask for JSON; on a parse failure, feed the error back once and retry."""
    reply = client.chat(system, user, json_mode=True, tag=tag)
    for attempt in range(retries + 1):
        try:
            return extract_json(reply)
        except (ValueError, json.JSONDecodeError) as exc:
            if attempt == retries:
                raise LLMError(f"[{tag}] model did not return valid JSON: {exc}\n---\n{reply[:800]}") from exc
            repair_prompt = (
                f"{user}\n\nYour previous answer could not be parsed as JSON ({exc}). "
                "Return ONLY the JSON object, with no prose and no code fences. "
                "Write LaTeX backslashes doubled inside JSON strings (\\\\frac)."
            )
            reply = client.chat(system, repair_prompt, json_mode=True, tag=f"{tag}.retry")
    raise AssertionError("unreachable")
