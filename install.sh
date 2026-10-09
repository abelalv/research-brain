#!/usr/bin/env bash
# research-brain installer — uses uv. Ubuntu 24.04 / 26.04, WSL2 and macOS.
#
#   bash install.sh                         # vault in ~/ResearchBrain, asks before each change
#   bash install.sh ~/Documents/Brain       # another vault location
#   bash install.sh --with-litellm          # also install the LiteLLM gateway (Bedrock, Vertex, Azure...)
#   bash install.sh --with-mcp              # MCP server: Claude Code / Claude Desktop use the vault's RAG as tools
#   bash install.sh --with-chroma           # ChromaDB as vector store for the RAG index (default: NumPy files)
#   bash install.sh --yes                   # accept every default (non-interactive)
#
# Environment overrides: RBRAIN_PYTHON (default 3.12), OLLAMA_HOST (remote/Windows Ollama).
#
# Steps:
#   1. uv              — reuse it if present, otherwise install it (no sudo, ~/.local/bin)
#   2. research-brain  — `uv tool install --editable` with a uv-managed Python (no system venv needed)
#   3. Ollama          — detect an existing server or binary FIRST (also Ollama for Windows seen
#                        from WSL); only offer to install when nothing is found
#   4. vault           — create it, or complete an existing one without touching your notes
#   5. models          — reuse the models you already have; pull only what is missing
#   6. connections     — optional: Claude (subscription or API), DeepSeek, OpenRouter, Gemini...
set -euo pipefail

VAULT="$HOME/ResearchBrain"
ASSUME_YES=0
WITH_LITELLM=0
WITH_MCP=0
WITH_CHROMA=0
for arg in "$@"; do
  case "$arg" in
    --yes|-y) ASSUME_YES=1 ;;
    --with-litellm) WITH_LITELLM=1 ;;
    --with-mcp) WITH_MCP=1 ;;
    --with-chroma) WITH_CHROMA=1 ;;
    -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
    -*) echo "Unknown option $arg"; exit 1 ;;
    *) VAULT="$arg" ;;
  esac
done
REPO="$(cd "$(dirname "$0")" && pwd)"
PYVER="${RBRAIN_PYTHON:-3.12}"

bold()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
ok()    { printf '  \033[32m✔\033[0m %s\n' "$*"; }
info()  { printf '  · %s\n' "$*"; }
warn()  { printf '  \033[33m!\033[0m %s\n' "$*"; }
ask()   { # ask "question" [default y|n]
  local def="${2:-n}" a
  if [ "$ASSUME_YES" = 1 ]; then [ "$def" = y ]; return; fi   # --yes: take the default
  read -r -p "  $1 [$( [ "$def" = y ] && echo Y/n || echo y/N )] " a || a=""
  a="${a:-$def}"; [[ "$a" =~ ^[Yy]$ ]]
}
is_wsl() { grep -qi microsoft /proc/version 2>/dev/null; }

# ----------------------------------------------------------------------------- 1. uv
bold "1/6 uv"
export PATH="$HOME/.local/bin:$PATH"
if command -v uv >/dev/null 2>&1; then
  ok "uv $(uv --version | awk '{print $2}') already installed ($(command -v uv))"
else
  if ask "uv is not installed. Install it now (official script, no sudo, into ~/.local/bin)?" y; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    ok "uv $(uv --version | awk '{print $2}') installed"
  else
    echo "uv is required. See https://docs.astral.sh/uv/ and run this script again."; exit 1
  fi
fi

# ----------------------------------------------------------------------------- 2. package
bold "2/6 Installing research-brain with uv (Python $PYVER managed by uv)"
WITH=()
if [ "$WITH_LITELLM" = 1 ]; then
  WITH+=(--with 'litellm!=1.82.7,!=1.82.8')   # 1.82.7/1.82.8 were a malicious release (Mar 2026)
  info "including the LiteLLM gateway"
fi
if [ "$WITH_MCP" = 1 ]; then WITH+=(--with 'mcp>=1.2'); info "including the MCP server"; fi
if [ "$WITH_CHROMA" = 1 ]; then WITH+=(--with 'chromadb>=0.5'); info "including ChromaDB"; fi
uv tool install --quiet --force --python "$PYVER" --editable "$REPO" "${WITH[@]}"
uv tool update-shell >/dev/null 2>&1 || true
BIN_DIR="$(uv tool dir --bin 2>/dev/null || echo "$HOME/.local/bin")"
export PATH="$BIN_DIR:$PATH"
ok "$(rbrain --version) → $(command -v rbrain)"
info "editable install: changes to the code and skills in $REPO take effect immediately"

# ----------------------------------------------------------------------------- 3. Ollama
bold "3/6 Ollama (local models)"
OLLAMA_URL="${OLLAMA_HOST:-http://localhost:11434}"
[[ "$OLLAMA_URL" == http* ]] || OLLAMA_URL="http://$OLLAMA_URL"
[[ "$OLLAMA_URL" =~ :[0-9]+$ ]] || OLLAMA_URL="$OLLAMA_URL:11434"
ollama_up() { curl -fsS --max-time 3 "$OLLAMA_URL/api/version" >/dev/null 2>&1; }
ollama_ver() { curl -fsS --max-time 3 "$OLLAMA_URL/api/version" 2>/dev/null | sed -E 's/.*"version"[[:space:]]*:[[:space:]]*"([^"]+)".*/\1/'; }
wait_ollama() { for _ in $(seq 1 15); do ollama_up && return 0; sleep 1; done; return 1; }

OLLAMA_OK=0
if ollama_up; then
  OLLAMA_OK=1
  ok "Ollama v$(ollama_ver) is already running at $OLLAMA_URL — nothing to install"
  if ! command -v ollama >/dev/null 2>&1; then
    info "no ollama binary here: the server lives elsewhere (e.g. Ollama for Windows seen from WSL). That is fine."
  fi
elif command -v ollama >/dev/null 2>&1; then
  ok "Ollama is installed ($(command -v ollama)) but not running"
  if command -v systemctl >/dev/null 2>&1 && systemctl list-unit-files ollama.service >/dev/null 2>&1 \
     && systemctl list-unit-files ollama.service | grep -q ollama; then
    if ask "Start the ollama service now (sudo systemctl start ollama)?" y; then
      sudo systemctl start ollama && wait_ollama && OLLAMA_OK=1
    fi
  elif ask "Start 'ollama serve' in the background now?" y; then
    nohup ollama serve >"$HOME/.ollama-serve.log" 2>&1 &
    wait_ollama && OLLAMA_OK=1
  fi
  [ "$OLLAMA_OK" = 1 ] && ok "Ollama is running" || warn "Ollama did not start; see 'rbrain ollama' later"
elif is_wsl && compgen -G "/mnt/c/Users/*/AppData/Local/Programs/Ollama/ollama*.exe" >/dev/null; then
  warn "Ollama is installed on the Windows side but is not reachable from WSL."
  info "Start the Ollama app on Windows. With mirrored networking (networkingMode=mirrored in .wslconfig)"
  info "it appears at localhost; otherwise export OLLAMA_HOST=<windows-ip>:11434. No second copy is needed."
  if ask "Install a separate Ollama inside WSL anyway?" n; then
    curl -fsSL https://ollama.com/install.sh | sh && wait_ollama && OLLAMA_OK=1
  fi
elif command -v snap >/dev/null 2>&1 && snap list ollama >/dev/null 2>&1; then
  warn "Ollama is installed as a snap but not running: sudo snap start ollama"
else
  info "Ollama was not found (no server at $OLLAMA_URL, no binary, no Windows install)."
  if ask "Install Ollama now with the official script (uses sudo)?" y; then
    curl -fsSL https://ollama.com/install.sh | sh
    wait_ollama && OLLAMA_OK=1
  else
    info "Skipped. You can still use Claude, DeepSeek or other APIs (step 6)."
  fi
fi

# ----------------------------------------------------------------------------- 4. vault
bold "4/6 Vault at $VAULT"
if [ -f "$VAULT/.rbrain/config.toml" ]; then
  ok "existing vault found — only missing template files are added; notes and config stay untouched"
fi
rbrain init "$VAULT" | head -1

# ----------------------------------------------------------------------------- 5. models
bold "5/6 Models"
if [ "$OLLAMA_OK" = 1 ]; then
  rbrain --vault "$VAULT" ollama || true
  echo
  if ask "Pick the default chat model from the models you ALREADY have (no download)?" y; then
    YES_FLAG=(); [ "$ASSUME_YES" = 1 ] && YES_FLAG=(--yes)
    rbrain --vault "$VAULT" connect ollama --name local --default "${YES_FLAG[@]}" || warn "skipped"
  fi
  if ask "Download the models this vault still needs (e.g. the bge-m3 embedding model)?" y; then
    rbrain --vault "$VAULT" ollama --pull-missing --yes || warn "some downloads failed; retry with: rbrain ollama --pull-missing"
  fi
else
  info "No local model server; skipping. Later: rbrain ollama  /  rbrain connect"
fi

# ----------------------------------------------------------------------------- 6. connections
bold "6/6 Other connections (optional)"
if command -v claude >/dev/null 2>&1; then
  ok "Claude Code found: the 'claude' profile uses your Claude subscription (no API key)"
elif ask "Install Claude Code (agent mode + Claude with your subscription)?" n; then
  curl -fsSL https://claude.ai/install.sh | bash || warn "Claude Code install failed"
fi
info "Available: Claude (subscription via Claude Code, or API key), DeepSeek, OpenRouter, Gemini,"
info "Groq, Mistral, OpenAI, Together, LM Studio, any OpenAI-compatible server, LiteLLM."
while [ "$ASSUME_YES" = 0 ] && ask "Connect a provider now?" n; do
  rbrain --vault "$VAULT" connect || warn "connection not saved"
done

bold "RAG index"
if [ "$WITH_CHROMA" = 1 ]; then
  sed -i 's/^backend = "numpy"/backend = "chroma"/' "$VAULT/.rbrain/config.toml" && ok "vector store: ChromaDB"
fi
rbrain --vault "$VAULT" rag index || warn "index not built yet; it is built automatically on the first question"
if [ "$WITH_MCP" = 1 ]; then
  rbrain --vault "$VAULT" mcp --install
fi

bold "Checking the setup"
rbrain --vault "$VAULT" doctor || warn "some checks failed — fix them with: rbrain connect   or   rbrain ollama"

cat <<EOF

Done.
  • Open $VAULT in Obsidian (Open folder as vault) and install the Dataview plugin.
  • Try:        rbrain ingest paper.pdf      rbrain explain "inverse problem"
  • Models:     rbrain connect --list        rbrain connect deepseek
  • RAG:        rbrain ask "..."  ·  rbrain chat  ·  rbrain rag search "..."
  • Agent mode: rbrain agent                 rbrain agent --backend deepseek
  • New shell needed? The command lives in $BIN_DIR
  • Uninstall:  uv tool uninstall research-brain   (your vault is never touched)
EOF
