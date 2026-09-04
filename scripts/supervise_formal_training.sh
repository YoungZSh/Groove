#!/usr/bin/env bash
# Keep the overnight Batch-8 visual-evidence run alive across a transient Ray/vLLM failure.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXPERIMENT_NAME="qwen35-4b-groove-visual-evidence-batch8-seq64-formal-v1"
CHECKPOINT_DIR="$PROJECT_ROOT/checkpoints/$EXPERIMENT_NAME"
LOG_FILE="$PROJECT_ROOT/outputs/logs/$EXPERIMENT_NAME.log"
SUPERVISOR_LOG="$PROJECT_ROOT/outputs/logs/$EXPERIMENT_NAME-supervisor.log"
REPORT_FILE="$PROJECT_ROOT/outputs/reports/qwen35-4b-batch8-seq64-latest.md"
TOKEN_DUMP_DIR="$PROJECT_ROOT/outputs/opsd-token-dumps/$EXPERIMENT_NAME"
TOKEN_REPORT_FILE="$PROJECT_ROOT/outputs/reports/qwen35-4b-opsd-token-credit-latest.md"
TARGET_STEPS=741

is_training_alive() {
  local pid args
  while IFS= read -r pid; do
    args="$(ps -p "$pid" -o args= 2>/dev/null || true)"
    if [[ "$args" == *"groove.verl_entrypoint"* && "$args" == *"trainer.experiment_name=$EXPERIMENT_NAME"* ]]; then
      return 0
    fi
  done < <(pgrep -x python || true)
  return 1
}

latest_step() {
  local tracker="$CHECKPOINT_DIR/latest_checkpointed_iteration.txt"
  if [[ -f "$tracker" ]]; then
    tr -dc '0-9' < "$tracker"
  fi
}

start_training() {
  export ANALYZER_BASE_URL="http://127.0.0.1:8002/v1"
  export ANALYZER_API_KEY="remote-qwen38"
  export ANALYZER_MODEL="Qwen3.8-27B"
  export ANALYZER_DISABLE_THINKING=true
  export ANALYZER_USE_VISION_TOOLS=true
  export ANALYZER_MAX_TOOL_ROUNDS=3
  export ANALYZER_TOOL_FEEDBACK_MAX_SIDE="${ANALYZER_TOOL_FEEDBACK_MAX_SIDE:-1024}"
  export ANALYZER_GROUNDING_URL="http://127.0.0.1:8011"
  export ANALYZER_OCR_URL="http://127.0.0.1:8012"
  export GROOVE_EVIDENCE_DIR="$PROJECT_ROOT/outputs/evidence-batch8-seq64-formal-v1"
  export GROOVE_REQUIRE_SLEEP_LEVEL_2=false
  export CUDA_VISIBLE_DEVICES=0,1
  export N_GPUS=2
  export OPSD_ENABLED=true
  export ROLLOUT_TENSOR_PARALLEL_SIZE=2
  export ROLLOUT_GPU_MEMORY_UTILIZATION=0.12
  export ROLLOUT_MAX_NUM_SEQS=64
  export TRAIN_BATCH_SIZE=8
  export ROLLOUT_N=8
  export PPO_MINI_BATCH_SIZE=64
  # Match Vision-OPD's 8k prompt policy: keep the released full-resolution
  # student image and rely only on Qwen's normal patch-alignment smart resize.
  export MAX_PROMPT_LENGTH=8192
  export STUDENT_IMAGE_MAX_PIXELS=null
  export STUDENT_IMAGE_PATCH_SIZE=16
  # Match Vision-OPD's full response budget so rare long rationales are not
  # truncated. Non-thinking prompts normally terminate far before this ceiling.
  export MAX_MODEL_LEN=9216
  export MAX_RESPONSE_LENGTH=1024
  export OPTIMIZER_IMPL=torch.optim
  export OPTIMIZER_NAME=AdamW
  export FUSED_ADAMW=false
  export ACTOR_PARAM_OFFLOAD=false
  export ACTOR_OPTIMIZER_OFFLOAD=false
  export REF_PARAM_OFFLOAD=false
  export ACTOR_ACTIVATION_OFFLOAD=false
  export TRAINING_FSDP_STRATEGY=fsdp
  export ACTOR_FSDP_OFFLOAD_POLICY=false
  export TOTAL_STEPS="$TARGET_STEPS"
  export SAVE_FREQ=10
  export TEST_FREQ=100
  export MAX_ACTOR_CKPT_TO_KEEP=2
  export OPSD_LOG_PROB_DUMP_DIR="$PROJECT_ROOT/outputs/opsd-token-dumps/$EXPERIMENT_NAME"
  export EXPERIMENT="$EXPERIMENT_NAME"
  export EXPERIMENT_NAME
  export CHECKPOINT_DIR
  export ROLLOUT_DATA_DIR="$PROJECT_ROOT/outputs/rollouts-batch8-seq64-formal-v1"

  set -o pipefail
  bash "$PROJECT_ROOT/scripts/run_groove.sh" 2>&1 | tee -a "$LOG_FILE"
}

write_progress_report() {
  /home/yzs/miniconda3/envs/vision-opd/bin/python \
    "$PROJECT_ROOT/scripts/summarize_training_progress.py" \
    --log "$LOG_FILE" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --evidence-dir "$PROJECT_ROOT/outputs/evidence-batch8-seq64-formal-v1" \
    --output "$REPORT_FILE" \
    >> "$SUPERVISOR_LOG" 2>&1
}

latest_token_dump_step() {
  local latest
  latest="$(find "$TOKEN_DUMP_DIR" -maxdepth 1 -type f -name '*.rank*.pt' -printf '%f\n' 2>/dev/null \
    | cut -d. -f1 | sort -n | tail -1)"
  printf '%s' "$latest"
}

write_token_report() {
  local step="$1"
  /home/yzs/miniconda3/envs/vision-opd/bin/python \
    "$PROJECT_ROOT/scripts/analyze_opd_tokens.py" \
    --dump-dir "$TOKEN_DUMP_DIR" \
    --rollouts-dir "$PROJECT_ROOT/outputs/rollouts-batch8-seq64-formal-v1" \
    --tokenizer "/root/siton-tmp/yzs/ckpts/Qwen3.5-4B" \
    --step "$step" \
    --output "$TOKEN_REPORT_FILE" \
    >> "$SUPERVISOR_LOG" 2>&1
}

last_reported_step=""
last_token_reported_step=""
while true; do
  current_step="$(latest_step)"
  if [[ -n "$current_step" && "$current_step" != "$last_reported_step" ]]; then
    if write_progress_report; then
      printf '%s refreshed report at step %s\n' "$(date --iso-8601=seconds)" "$current_step" >> "$SUPERVISOR_LOG"
      last_reported_step="$current_step"
    fi
  fi
  token_step="$(latest_token_dump_step)"
  if [[ -n "$token_step" && "$token_step" != "$last_token_reported_step" ]]; then
    if write_token_report "$token_step"; then
      printf '%s refreshed OPD token report at step %s\n' \
        "$(date --iso-8601=seconds)" "$token_step" >> "$SUPERVISOR_LOG"
      last_token_reported_step="$token_step"
    fi
  fi
  if [[ -n "$current_step" && "$current_step" -ge "$TARGET_STEPS" ]]; then
    printf '%s completed at step %s\n' "$(date --iso-8601=seconds)" "$current_step" >> "$SUPERVISOR_LOG"
    exit 0
  fi

  if is_training_alive; then
    sleep 60
    continue
  fi

  printf '%s restarting from checkpoint step %s\n' "$(date --iso-8601=seconds)" "${current_step:-0}" >> "$SUPERVISOR_LOG"
  if ! start_training; then
    printf '%s training exited; retrying after 60 seconds\n' "$(date --iso-8601=seconds)" >> "$SUPERVISOR_LOG"
    sleep 60
  fi
done
