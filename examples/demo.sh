#!/usr/bin/env bash
# Build a complete demo vault WITHOUT any model, using canned responses in
# examples/fake-responses/. Open the resulting folder in Obsidian to see what the
# skills produce (graph view, Dashboard, concept page, ideas, argument...).
#
#   bash examples/demo.sh            # vault in /tmp/rbrain-demo
#   bash examples/demo.sh ~/Demo     # vault elsewhere
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
VAULT="${1:-/tmp/rbrain-demo}"
export RBRAIN_FAKE_DIR="$HERE/fake-responses"

rbrain init "$VAULT" --force --no-default

# Point every skill at the "fake" provider (canned answers, hashed embeddings).
cat > "$VAULT/.rbrain/config.toml" <<'EOF'
[vault]
language = "English"
[llm]
provider = "fake"
model = "demo"
[embeddings]
enabled = true
provider = "fake"
model = "hash"
EOF

rb() { rbrain --vault "$VAULT" "$@"; }

echo; echo "== 1. Ingest a source (summary + idea segmentation)"
rb ingest "$HERE/sources/pinn-sir-inverse-problem.md"

echo; echo "== 2. Explain a concept"
rb explain "physics-informed neural network"

echo; echo "== 3. Construct new ideas"
rb synthesize "[[a-pinn-treats-the-transmission-rate-as-a-trainable-parameter]]" \
              "[[can-pinns-estimate-spatially-varying-transmission-on-contact-networks]]" \
              -q "graph neural networks for spatial epidemic spread"

echo; echo "== 4. Deductive reasoning with symbolic check and repair"
rb deduce \
  -p "If the residual vanishes on the whole interval and the initial conditions hold, the network solves the IVP" \
  -p "If the network solves the IVP and the IVP has a unique solution, the network equals the true trajectory" \
  -p "Training drove the residual to zero" \
  -p "The initial conditions are satisfied" \
  -g "The trained PINN equals the true SIR trajectory"

echo; echo "== 5. RAG: index passages, inspect retrieval, ask with cited passages"
rb rag index
rb rag stats
rb rag search "identifiability of the reporting fraction" -k 3
rb ask "What are the main limitations of PINNs for estimating epidemic parameters?" --save

echo; echo "== 6. Lint (health check + research agenda)"
rb lint --llm

echo; echo "== 7. Pure logic tools (no LLM)"
rb logic check "Res0 & IC -> Solves" "Solves & Uniq -> Equal" "Res0" "IC" --goal "Equal"
rb logic table "(P -> Q) <-> (~Q -> ~P)"

echo; echo "Demo vault ready: $VAULT  — open it in Obsidian."
