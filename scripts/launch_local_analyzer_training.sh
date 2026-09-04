#!/usr/bin/env bash
# Start the local Analyzer on GPU 0, wait for readiness, then train on GPU 1.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${RUN_LOG_DIR:-$PROJECT_ROOT/outputs/logs}"
ANALYZER_HOST="${ANALYZER_HOST:-127.0.0.1}"
ANALYZER_PORT="${ANALYZER_PORT:-8001}"
ANALYZER_MODEL_NAME="${ANALYZER_MODEL_NAME:-qwen35-35b-a3b-fp8-analyzer}"
ANALYZER_API_KEY="${ANALYZER_API_KEY:-local-qwen-analyzer}"
ANALYZER_HEALTH_URL="http://${ANALYZER_HOST}:${ANALYZER_PORT}/health"
ANALYZER_LOG="${ANALYZER_LOG:-$LOG_DIR/qwen35-analyzer.log}"
READY_TIMEOUT_SECONDS="${ANALYZER_READY_TIMEOUT_SECONDS:-1800}"

mkdir -p "$LOG_DIR"

if ! curl --fail --silent --max-time 3 "$ANALYZER_HEALTH_URL" >/dev/null; then
  nohup env \
    ANALYZER_HOST="$ANALYZER_HOST" \
    ANALYZER_PORT="$ANALYZER_PORT" \
    ANALYZER_MODEL_NAME="$ANALYZER_MODEL_NAME" \
    ANALYZER_API_KEY="$ANALYZER_API_KEY" \
    "$PROJECT_ROOT/scripts/serve_qwen35_analyzer.sh" >"$ANALYZER_LOG" 2>&1 &
  echo $! >"$LOG_DIR/qwen35-analyzer.pid"
fi

deadline=$((SECONDS + READY_TIMEOUT_SECONDS))
until curl --fail --silent --max-time 3 "$ANALYZER_HEALTH_URL" >/dev/null; do
  if (( SECONDS >= deadline )); then
    echo "Analyzer did not become ready; inspect $ANALYZER_LOG" >&2
    exit 1
  fi
  sleep 5
done

export ANALYZER_BASE_URL="http://${ANALYZER_HOST}:${ANALYZER_PORT}/v1"
export ANALYZER_API_KEY
export ANALYZER_MODEL="$ANALYZER_MODEL_NAME"
export ANALYZER_DISABLE_THINKING=true
export ANALYZER_TOOL_FEEDBACK_MAX_SIDE="${ANALYZER_TOOL_FEEDBACK_MAX_SIDE:-1024}"

# The Analyzer permanently occupies GPU 0.  VERL's hybrid worker gets GPU 1 and
# still uses sleep level 2 plus CPU offload for rollout/training phase changes.
export CUDA_VISIBLE_DEVICES="${TRAIN_GPU_ID:-1}"
export N_GPUS=1
export OPSD_ENABLED=true
export GROUNDING_DINO_DEVICE=cpu
export ANALYZER_USE_VISION_TOOLS="${ANALYZER_USE_VISION_TOOLS:-true}"
# OCR uses the Analyzer's GPU during evidence construction.  GPU 1 remains
# exclusively available to the colocated actor/rollout worker.
export ANALYZER_OCR_GPU_ID="${ANALYZER_OCR_GPU_ID:-0}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
export ROLLOUT_N="${ROLLOUT_N:-8}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
# The current Qwen3.5 DeltaNet/vLLM path decodes this visual batch almost
# serially. 128 tokens comfortably fits a concise VQA rationale + FINAL line
# while making the sustained on-policy run materially more productive.
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-128}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
export ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-8}"
# The colocated FSDP actor/ref temporarily holds roughly 52 GiB during rollout
# engine creation.  Leave vLLM 28% of the A100 at startup; sleep level 2 and
# free_cache_engine release it again before the optimization phase.
# Keep enough headroom while the current-policy FSDP state is copied into the
# rollout worker.  At 8k context, 16% is sufficient for the 4B model + cache.
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.16}"
# Prefer full-precision AdamW.  The FSDP2 CPU offload policy selected below
# keeps its FP32 moments off the training GPU rather than quantizing them.
export OPTIMIZER_IMPL="${OPTIMIZER_IMPL:-torch.optim}"
export OPTIMIZER_NAME="${OPTIMIZER_NAME:-AdamW}"
export FUSED_ADAMW="${FUSED_ADAMW:-false}"
export ACTOR_ACTIVATION_OFFLOAD="${ACTOR_ACTIVATION_OFFLOAD:-true}"
export TRAINING_FSDP_STRATEGY="${TRAINING_FSDP_STRATEGY:-fsdp}"
if [[ "$TRAINING_FSDP_STRATEGY" == "fsdp2" ]]; then
  export ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-false}"
  export ACTOR_OPTIMIZER_OFFLOAD="${ACTOR_OPTIMIZER_OFFLOAD:-false}"
  export ACTOR_FSDP_OFFLOAD_POLICY="${ACTOR_FSDP_OFFLOAD_POLICY:-true}"
else
  # FSDP1 releases model/optimizer state between rollout and update.  The
  # synchronous activation offloader additionally keeps the full-precision
  # AdamW update below the single-A100 memory limit.
  export ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-true}"
  export ACTOR_OPTIMIZER_OFFLOAD="${ACTOR_OPTIMIZER_OFFLOAD:-true}"
  export ACTOR_FSDP_OFFLOAD_POLICY="${ACTOR_FSDP_OFFLOAD_POLICY:-false}"
fi
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-false}"
export TEST_FREQ="${TEST_FREQ:-500}"
export TOTAL_STEPS="${TOTAL_STEPS:-150}"
exec "$PROJECT_ROOT/scripts/run_groove.sh" "$@"
