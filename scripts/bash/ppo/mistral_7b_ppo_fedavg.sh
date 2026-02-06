#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

NPROC="${NPROC:-1}"

# Environment requirements for PPO runs:
# - HF_TOKEN or HUGGINGFACE_HUB_TOKEN: access to HF models/datasets
# - WANDB_API_KEY: enable wandb logging (optional; set WANDB_MODE=offline to disable)
# - OPENAI_API_KEY: required only if evaluator arbiter uses OpenAI/DeepInfra (see config)
# - PPO_PREFIX: base directory for models/checkpoints/logs (default is repo root)

if [ -z "${HF_TOKEN:-}" ] && [ -z "${HUGGINGFACE_HUB_TOKEN:-}" ]; then
  echo "[warn] HF_TOKEN/HUGGINGFACE_HUB_TOKEN not set. HF model/dataset downloads may fail."
fi

if [ -z "${WANDB_API_KEY:-}" ]; then
  echo "[warn] WANDB_API_KEY not set. Set WANDB_MODE=offline to avoid auth errors."
fi

if grep -q "base_url: https://api.deepinfra.com/v1/openai" "scripts/configs/ppo/tldr/mistral_7b_ppo_fedavg.yaml" 2>/dev/null; then
  if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "[warn] OPENAI_API_KEY not set, but config uses remote arbiter."
  fi
fi

tune run --nproc_per_node "$NPROC" src/rlhf/recipes/ppo.py --config "scripts/configs/ppo/tldr/mistral_7b_ppo_fedavg.yaml" "$@"
