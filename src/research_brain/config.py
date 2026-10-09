"""Configuration loading and vault discovery.

Each vault carries its own configuration in ``<vault>/.rbrain/config.toml``. The vault is
found, in order of priority, from: the ``--vault`` flag, the ``RBRAIN_VAULT`` environment
variable, the current directory (or any parent) containing ``.rbrain/config.toml``, and
finally the default vault recorded by ``rbrain init`` in ``~/.config/research-brain/``.

API keys never live in the vault: they come from environment variables or from
``~/.config/research-brain/secrets.env`` (written by ``rbrain connect``, mode 600).
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from . import providers

CONFIG_DIRNAME = ".rbrain"
CONFIG_FILENAME = "config.toml"
USER_CONFIG = Path.home() / ".config" / "research-brain" / "default_vault"

DEFAULT_CONFIG_TOML = """\
# research-brain configuration (one file per vault).
# Easiest way to change models:   rbrain connect          (interactive)
# Check everything:               rbrain doctor --live
#
# Providers: ollama | lmstudio | claude_code | anthropic | deepseek | openrouter | gemini
#            | groq | mistral | openai | together | openai_compat | litellm
# API keys are NOT stored here: they live in ~/.config/research-brain/secrets.env or in
# your environment (DEEPSEEK_API_KEY, ANTHROPIC_API_KEY, OPENROUTER_API_KEY, ...).

[vault]
language = "English"      # language of the notes the LLM writes
max_atoms_per_chunk = 10  # cap for idea segmentation per text chunk
chunk_chars = 6000        # long sources are split into chunks of this size

# ---------------------------------------------------------------- default LLM
[llm]
# default_profile = "deepseek"   # use a profile below as the default instead of these lines
provider = "ollama"
model = "qwen2.5:7b-instruct-q4_K_M"
temperature = 0.3
num_ctx = 16384           # Ollama only. Ollama's default context is small and truncates silently.
max_tokens = 4096
timeout = 900             # seconds; local models on CPU can be slow
# think = false           # Ollama only: disable "thinking" on hybrid models such as qwen3

# ------------------------------------------------- named profiles (rbrain connect adds more)
# Use one for a single command with:  rbrain --profile claude synthesize ...
[llm.profiles.reasoning]
provider = "ollama"
model = "deepseek-r1:14b"
temperature = 0.2
num_ctx = 16384
max_tokens = 6000

[llm.profiles.claude]
provider = "claude_code"   # your Claude Pro/Max subscription through Claude Code, no API key
model = "sonnet"           # sonnet | opus | haiku | a full model name

# [llm.profiles.claude-api]
# provider = "anthropic"   # pay-per-use API key (ANTHROPIC_API_KEY)
# model = "claude-sonnet-5-5"

# [llm.profiles.deepseek]
# provider = "deepseek"    # DEEPSEEK_API_KEY
# model = "deepseek-flash" # or deepseek-v4-pro

# Which profile each skill uses (skills not listed use the default).
[skill_profiles]
deductive-reasoning = "reasoning"
# synthesize-ideas = "claude"

# ------------------------------------------------------------- embeddings
# Hybrid search = BM25 (always on) + embeddings (optional). bge-m3 is multilingual
# (good for mixed English/Spanish notes); nomic-embed-text is lighter.
[embeddings]
enabled = true
provider = "ollama"
model = "bge-m3"
# query_prefix / document_prefix are filled automatically for nomic-embed-text and e5 models

[search]
top_k = 8

# -------------------------------------------------------------------- RAG
# Passage-level retrieval over every note AND the full text of every source.
# rbrain rag index | rbrain rag search "..." | rbrain ask "..." | rbrain chat
[rag]
backend = "numpy"         # numpy (no extra dependency) | chroma (install the chromadb extra)
chunk_chars = 1200        # passage size; equations and code blocks are never split
overlap_chars = 200       # whole paragraphs/sentences repeated between neighbouring passages
top_k = 8                 # passages given to the model
scope = "all"             # all | wiki (your notes) | sources (raw papers) for ask/chat
max_context_chars = 12000 # budget for passages in the prompt; raise it for 1M-context cloud models
max_per_note = 3          # at most this many passages from the same note
diversity = 0.3           # 0 = pure relevance, higher = less redundant passages (MMR)
auto_update = true        # re-index changed notes automatically before each question
expand_queries = false    # LLM writes 3 query variants (incl. English<->Spanish); slower, better recall
rerank = false            # LLM re-scores the candidates; slower, better precision
"""


@dataclass
class LLMProfile:
    """Connection settings for one model."""

    provider: str = "ollama"
    model: str = ""
    base_url: str = ""
    api_key_env: str = ""
    temperature: float = 0.3
    num_ctx: int = 16384
    max_tokens: int = 4096
    timeout: int = 900
    think: bool | None = None
    json_mode: bool | None = None  # None = provider default
    command: str = ""              # claude_code: path to the `claude` executable

    def __post_init__(self) -> None:
        spec = providers.get(self.provider)  # raises on unknown providers
        if not self.base_url:
            self.base_url = providers.ollama_url() if spec.api == "ollama" else spec.base_url
        if not self.api_key_env:
            self.api_key_env = spec.api_key_env
        if not self.model:
            self.model = spec.default_model
        if self.json_mode is None and spec.api != "ollama":
            self.json_mode = spec.json_mode

    @property
    def spec(self) -> providers.Provider:
        return providers.get(self.provider)

    @classmethod
    def from_dict(cls, data: dict) -> "LLMProfile":
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class EmbeddingConfig:
    enabled: bool = False
    provider: str = "ollama"
    model: str = "bge-m3"
    base_url: str = ""
    api_key_env: str = ""
    query_prefix: str = ""
    document_prefix: str = ""

    def as_profile(self) -> LLMProfile:
        return LLMProfile(provider=self.provider, model=self.model, base_url=self.base_url,
                          api_key_env=self.api_key_env)


@dataclass
class RagConfig:
    backend: str = "numpy"
    chunk_chars: int = 1200
    overlap_chars: int = 200
    top_k: int = 8
    scope: str = "all"
    max_context_chars: int = 12000
    max_per_note: int = 3
    diversity: float = 0.3
    auto_update: bool = True
    expand_queries: bool = False
    rerank: bool = False

    def __post_init__(self) -> None:
        if self.backend not in {"numpy", "chroma"}:
            raise ValueError(f"[rag] backend must be 'numpy' or 'chroma', not '{self.backend}'")
        if self.scope not in {"all", "wiki", "sources", "inbox"}:
            raise ValueError(f"[rag] scope must be all, wiki or sources, not '{self.scope}'")


@dataclass
class Config:
    vault: Path
    language: str = "English"
    max_atoms_per_chunk: int = 10
    chunk_chars: int = 6000
    llm: LLMProfile = field(default_factory=LLMProfile)
    profiles: dict[str, LLMProfile] = field(default_factory=dict)
    skill_profiles: dict[str, str] = field(default_factory=dict)
    embeddings: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    top_k: int = 8
    rag: RagConfig = field(default_factory=RagConfig)
    forced_profile: str | None = None  # set by the --profile CLI flag

    def profile_for(self, skill: str) -> LLMProfile:
        """Return the model profile that a given skill should use."""
        name = self.forced_profile or self.skill_profiles.get(skill)
        if name is None or name == "default":
            return self.llm
        if name not in self.profiles:
            raise KeyError(f"Profile '{name}' is not defined under [llm.profiles] in config.toml "
                           f"(available: {', '.join(self.profiles) or 'none'})")
        return self.profiles[name]

    def all_profiles(self) -> dict[str, LLMProfile]:
        return {"default": self.llm, **self.profiles}


def config_path(vault: Path) -> Path:
    return vault / CONFIG_DIRNAME / CONFIG_FILENAME


def find_vault(explicit: str | os.PathLike | None = None) -> Path:
    """Locate the vault root using the priority order described in the module docstring."""
    if explicit:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get("RBRAIN_VAULT")
    if env:
        return Path(env).expanduser().resolve()
    here = Path.cwd().resolve()
    for candidate in (here, *here.parents):
        if config_path(candidate).exists():
            return candidate
    if USER_CONFIG.exists():
        return Path(USER_CONFIG.read_text(encoding="utf-8").strip()).expanduser().resolve()
    raise FileNotFoundError(
        "No vault found. Run `rbrain init ~/ResearchBrain` or pass --vault PATH."
    )


def load_config(vault: Path) -> Config:
    """Read ``<vault>/.rbrain/config.toml`` into a :class:`Config` (and load saved API keys)."""
    providers.load_secrets()
    path = config_path(vault)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Is {vault} a research-brain vault?")
    data = tomllib.loads(path.read_text(encoding="utf-8"))

    vault_cfg = data.get("vault", {})
    llm_cfg = dict(data.get("llm", {}))
    profiles_cfg = llm_cfg.pop("profiles", {})
    default_name = llm_cfg.pop("default_profile", None)
    emb_cfg = data.get("embeddings", {})

    profiles = {name: LLMProfile.from_dict(p) for name, p in profiles_cfg.items()}
    if default_name:
        if default_name not in profiles:
            raise KeyError(f"[llm] default_profile = '{default_name}' but no [llm.profiles.{default_name}]")
        default = profiles[default_name]
    else:
        default = LLMProfile.from_dict(llm_cfg)

    return Config(
        vault=vault,
        language=vault_cfg.get("language", "English"),
        max_atoms_per_chunk=int(vault_cfg.get("max_atoms_per_chunk", 10)),
        chunk_chars=int(vault_cfg.get("chunk_chars", 6000)),
        llm=default,
        profiles=profiles,
        skill_profiles=dict(data.get("skill_profiles", {})),
        embeddings=EmbeddingConfig(**{k: v for k, v in emb_cfg.items()
                                      if k in EmbeddingConfig.__dataclass_fields__}),
        top_k=int(data.get("search", {}).get("top_k", 8)),
        rag=RagConfig(**{k: v for k, v in data.get("rag", {}).items() if k in RagConfig.__dataclass_fields__}),
    )
