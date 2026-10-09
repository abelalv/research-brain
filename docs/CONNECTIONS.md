# Connecting models

research-brain can use a model in **two modes**:

- **Skill mode** (`rbrain ingest`, `segment`, `synthesize`, `explain`, `deduce`, `ask`, `lint`):
  the CLI sends the skill to a model and writes the notes itself. Any connection below works.
- **Agent mode** (`rbrain agent`): Claude Code opens in the vault, reads `CLAUDE.md` and the
  skills, and edits notes as an agent. It needs a model that speaks the Anthropic API:
  Claude, DeepSeek, or a local model through Ollama.

The quickest way to set up any connection is the wizard:

```bash
rbrain connect            # menu with the status of every option
rbrain connect --list     # what is configured and whether it is reachable
rbrain connect --test     # same, plus a one-line test prompt to each model
```

## 1. The options at a glance

| Connection | `provider` | Key needed | Best for |
|---|---|---|---|
| Ollama | `ollama` | no | private, free, offline; daily ingestion and segmentation |
| LM Studio | `lmstudio` | no | local models with a GUI |
| **Claude with your subscription** | `claude_code` | **no** (Pro/Max login) | synthesis and deduction with a top model at no extra cost |
| Claude API | `anthropic` | `ANTHROPIC_API_KEY` | pay-per-use Claude, batch work |
| DeepSeek | `deepseek` | `DEEPSEEK_API_KEY` | strong and cheap; JSON mode; 1M context |
| OpenRouter | `openrouter` | `OPENROUTER_API_KEY` | hundreds of models with one key |
| Gemini | `gemini` | `GEMINI_API_KEY` | long documents; embeddings |
| Groq | `groq` | `GROQ_API_KEY` | very fast open models |
| Mistral | `mistral` | `MISTRAL_API_KEY` | European provider; embeddings |
| OpenAI | `openai` | `OPENAI_API_KEY` | GPT models; embeddings |
| Together | `together` | `TOGETHER_API_KEY` | open models; embeddings |
| Any OpenAI-compatible server | `openai_compat` | optional | vLLM, llama.cpp server, a university GPU server |
| LiteLLM gateway | `litellm` | provider-specific | Bedrock, Vertex AI, Azure OpenAI, Cohere and 100+ more |

Model names change frequently, so the wizard asks each provider for its **current** model
list instead of relying on names written in this documentation.

## 2. Where API keys live

Keys are **never** written into the vault (which you may sync or commit to git). `rbrain
connect` stores them in `~/.config/research-brain/secrets.env` with permissions `600`, and
every `rbrain` command loads that file automatically. A variable already set in your shell
always wins, so `export DEEPSEEK_API_KEY=...` in `~/.bashrc` keeps working.

## 3. Recipes

### Claude with your Pro/Max subscription (no API key)

```bash
curl -fsSL https://claude.ai/install.sh | bash     # once; then run `claude` and log in
rbrain connect claude_code --model sonnet --skills synthesize-ideas,deductive-reasoning
```

How it works: for each skill call, `rbrain` runs `claude -p` headless from an empty
temporary directory, replaces Claude Code's system prompt with the skill, disables every
tool (`--tools ""`), and reads the answer from the JSON output. `ANTHROPIC_API_KEY` is
removed from the child environment so the subscription login is used. Each call counts
against your plan's usage limits, so assign it to the skills where quality matters most
(synthesis, deduction) and keep bulk work local. Models: `sonnet`, `opus`, `haiku` or a full
model name.

### Claude API (Console key)

```bash
rbrain connect anthropic                 # asks for the key, lists the available models
```

### DeepSeek

```bash
rbrain connect deepseek --default        # deepseek-flash by default; deepseek-v4-pro is larger
```

### Hybrid: local embeddings + DeepSeek generation

```bash
rbrain connect deepseek --default
rbrain connect ollama --model bge-m3 --embeddings
```

Search stays private (embeddings computed locally); only the notes involved in each
command are sent to DeepSeek.

### Several providers, each where it is best

```bash
rbrain connect ollama --name local --default                       # bulk work, private
rbrain connect claude_code --skills synthesize-ideas               # creative synthesis
rbrain connect deepseek --skills deductive-reasoning,query-brain   # long-context reasoning
rbrain --profile local explain "identifiability"                   # override for one command
```

### OpenRouter, Gemini, Groq, Mistral, OpenAI, Together

```bash
rbrain connect openrouter        # then pick e.g. a 'provider/model' id from the live list
rbrain connect gemini
```

### A server at the university or in the lab (vLLM, llama.cpp, TGI)

```bash
rbrain connect openai_compat --base-url http://gpu-server:8000/v1 --model Qwen2.5-32B-Instruct
```

### LiteLLM (Bedrock, Vertex AI, Azure OpenAI, Cohere...)

```bash
bash install.sh --with-litellm           # or: uv tool install --editable . --with 'litellm!=1.82.7,!=1.82.8'
rbrain connect litellm --model vertex_ai/<model-id>
```

LiteLLM reads each provider's usual credentials (AWS, Google Cloud, Azure variables).
Versions 1.82.7 and 1.82.8 of LiteLLM were a malicious supply-chain release in March 2026;
they are excluded, and `uv.lock` pins a verified version with hashes. Use LiteLLM only when
a provider is not covered natively — every connection above works without it.

## 4. Ollama: detection and models

```bash
rbrain ollama                 # server, version, installed models, models this vault needs
rbrain ollama --pull-missing  # download only what is missing
rbrain ollama --pull qwen3:4b
```

Ollama is found through its API first (`OLLAMA_HOST`, default `localhost:11434`), then
through its binary. On WSL, an Ollama installed **on Windows** is used directly — with
mirrored networking it appears at `localhost`; otherwise set
`export OLLAMA_HOST=<windows-ip>:11434`. There is no need to install a second copy inside
WSL. Downloads go through the API, so they also work when the `ollama` command is not on
the WSL side. `rbrain connect ollama` offers the models you already have before suggesting
any download.

## 5. Agent mode with other models

```bash
rbrain agent                                   # Claude Code + your Claude login
rbrain agent --backend deepseek                # DeepSeek's Anthropic-compatible endpoint
rbrain agent --backend deepseek --model deepseek-v4-pro
rbrain agent --backend ollama --model qwen3-coder
rbrain agent --backend custom --base-url https://host/anthropic --token-env MY_TOKEN --model my-model
rbrain agent --backend deepseek --print-env    # print the export lines instead of launching
rbrain agent -- --permission-mode plan         # arguments after -- go to claude
```

`rbrain agent` sets the variables documented by each provider for Claude Code
(`ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_MODEL` and the default
opus/sonnet/haiku/subagent models) and starts `claude` inside the vault. `--print-env` shows
secrets as `$VARIABLE` references, never as values.

With Ollama, keep in mind that Claude Code sends long prompts and depends on tool calling:
models of 14B parameters or more with good tool use work best, and the Ollama server needs a
large context window (for example `OLLAMA_CONTEXT_LENGTH=32768`). On an 8 GB GPU, agent mode
is usually more pleasant with the DeepSeek backend, while skill mode runs well locally.

## 6. Checking everything

```bash
rbrain doctor          # static checks: servers, models, keys, logins
rbrain doctor --live   # plus a one-line prompt to every profile in use
```

Profiles that no skill uses are reported as *optional*, so an unused Claude API profile
without a key does not make the check fail.
