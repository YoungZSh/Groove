#!/usr/bin/env bash
# GRPO-only ablation: disable online OPSD/evidence and use a 16-prompt batch.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

export OPSD_ENABLED=false
export TRAIN_BATCH_SIZE=16
export ROLLOUT_N="${ROLLOUT_N:-8}"
export PPO_MINI_BATCH_SIZE="$((TRAIN_BATCH_SIZE * ROLLOUT_N))"
# Full-resolution visual prompts in the prepared split can exceed 4096 tokens
# after multimodal chat-template expansion (observed maximum: 4160).
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"

exec "$PROJECT_ROOT/TMP/scripts/run_groove.sh" "$@"
