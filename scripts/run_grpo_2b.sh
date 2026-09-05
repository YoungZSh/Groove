#!/usr/bin/env bash
# Pure-GRPO Qwen3.5-2B run on the screened 2.2K visual-QA split.
set -euo pipefail

# Generate reasoning as ordinary response text, without a native think prefill.
# The reasoning/answer prompt and template are resolved by the entrypoint.
for argument in "$@"; do
  if [[ "$argument" == *"enable_thinking="* && "$argument" != *"enable_thinking=false" ]]; then
    echo "Qwen3.5 thinking must remain disabled for this experiment: $argument" >&2
    exit 2
  fi
done

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data/deepeyes_vstar_grpo_2200_seed20260904"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen35-2b-grpo-reasoning-answer-seed22}"

export MODEL_PATH="/root/siton-tmp/yzs/ckpts/Qwen3.5-2B"
export PREPARE_DATA=false
export DATA_OUTPUT_DIR="$DATA_DIR"
export TRAIN_FILE="$DATA_DIR/train.parquet"
export TEST_FILE="$DATA_DIR/validation.parquet"
export SEED=22

# Keep the full training history locally without requiring network access or a
# W&B login. The offline run can be synced after training completes.
export TRAINER_LOGGER='["console","wandb"]'
export WANDB_MODE=offline
export WANDB_PROJECT=groove-visual-evidence
export WANDB_NAME="$EXPERIMENT_NAME"
export WANDB_DIR="$PROJECT_ROOT/outputs/wandb"

export CUDA_VISIBLE_DEVICES=0,1
export N_GPUS=2
export OPSD_ENABLED=false
export GROOVE_REQUIRE_SLEEP_LEVEL_2=false

export TRAIN_BATCH_SIZE=16
export ROLLOUT_N=8
export PPO_MINI_BATCH_SIZE=16
# Pool valid response tokens across the batch. With the severe-repetition reward
# gate below, long failed trajectories receive proportionally more gradient.
export LOSS_AGG_MODE="${LOSS_AGG_MODE:-token-mean}"
# Dynamic micro-batching for actor, reference, and log-prob passes. The first
# end-to-end step used only ~19GB/GPU at 9K, so 32K materially improves GPU
# occupancy while retaining ample room on the two 80GB cards.
export ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU="${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU:-32768}"
# The selected split's real multimodal prompts are far below this bound. Keep
# a modest padding ceiling for the non-packed Qwen3.5 path without resizing or
# cropping the original images.
export MAX_PROMPT_LENGTH=2048
export MAX_RESPONSE_LENGTH=1024
export MAX_MODEL_LEN=9216
export ENABLE_THINKING=false
export STUDENT_RESPONSE_FORMAT=reasoning_answer
export STUDENT_IMAGE_MAX_PIXELS=null
export STUDENT_IMAGE_PATCH_SIZE=16

# VERL 0.9 passes packed cu_seqlens/seq_idx through Qwen3.5's Gated DeltaNet
# and causal-convolution layers, preventing cross-sample state leakage.
export MODEL_USE_REMOVE_PADDING=true

# Each DP rollout replica receives about 64 completions per training step.
export ROLLOUT_TENSOR_PARALLEL_SIZE=1
export ROLLOUT_MAX_NUM_SEQS=64
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=32768
export ROLLOUT_GPU_MEMORY_UTILIZATION=0.45
export ROLLOUT_ENFORCE_EAGER=true

# The 2B model and optimizer fit comfortably on two 80GB GPUs. Avoid CPU
# transfers between rollout and update phases to improve step time.
export TRAINING_FSDP_STRATEGY=fsdp
export ACTOR_PARAM_OFFLOAD=false
export ACTOR_OPTIMIZER_OFFLOAD=false
export REF_PARAM_OFFLOAD=false
export ACTOR_ACTIVATION_OFFLOAD=false
export ACTOR_FSDP_OFFLOAD_POLICY=false
export OPTIMIZER_IMPL=torch.optim
export OPTIMIZER_NAME=AdamW
export FUSED_ADAMW=true
export REFERENCE_KL_COEF=0.01

# Keep semantic judging independent, then subtract a 0.2 DeepEyes-style
# negative format penalty while retaining 1.0 as the maximum reward.
export CUSTOM_REWARD_FUNCTION_PATH="$PROJECT_ROOT/src/groove/deepeyes_reward.py"
export CUSTOM_REWARD_FUNCTION_NAME=compute_score
export REWARD_MANAGER_NAME=naive
# VERL 0.9's reward worker is an async Ray actor: concurrent trajectory calls
# share its executor, so one process can keep many remote Judge requests in
# flight without the instability and startup cost of dozens of worker actors.
export REWARD_NUM_WORKERS=1
export ANSWER_REWARD_WEIGHT=1.0
export FORMAT_REWARD_WEIGHT=0.2
export DEEPEYES_JUDGE_BASE_URL=http://127.0.0.1:8002/v1
export DEEPEYES_JUDGE_API_KEY="${DEEPEYES_JUDGE_API_KEY:-remote-qwen38}"
export DEEPEYES_JUDGE_MODEL=Qwen3.8-27B
export DEEPEYES_JUDGE_CONCURRENCY=128
export DEEPEYES_JUDGE_TIMEOUT_SECONDS=180
export DEEPEYES_JUDGE_MAX_RETRIES=5
export DEEPEYES_REPETITION_ZERO_REWARD="${DEEPEYES_REPETITION_ZERO_REWARD:-true}"
export DEEPEYES_REPETITION_MIN_REPEATS="${DEEPEYES_REPETITION_MIN_REPEATS:-4}"
export DEEPEYES_REPETITION_MIN_TOTAL_CHARACTERS="${DEEPEYES_REPETITION_MIN_TOTAL_CHARACTERS:-80}"
export DEEPEYES_REPETITION_MIN_PERIOD="${DEEPEYES_REPETITION_MIN_PERIOD:-1}"
export DEEPEYES_REPETITION_MAX_PERIOD="${DEEPEYES_REPETITION_MAX_PERIOD:-1024}"
export DEEPEYES_REPETITION_SAMPLE_LENGTH="${DEEPEYES_REPETITION_SAMPLE_LENGTH:-16}"
export DEEPEYES_REPETITION_SAMPLE_INTERVAL="${DEEPEYES_REPETITION_SAMPLE_INTERVAL:-32}"

export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
export TOTAL_STEPS="${TOTAL_STEPS:-null}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-true}"
export TEST_FREQ="${TEST_FREQ:-10}"
export SAVE_FREQ="${SAVE_FREQ:-10}"
export MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-2}"
export EXPERIMENT_NAME
export CHECKPOINT_DIR="$PROJECT_ROOT/checkpoints/$EXPERIMENT_NAME"
export ROLLOUT_DATA_DIR="$PROJECT_ROOT/outputs/rollouts/$EXPERIMENT_NAME"

exec "$PROJECT_ROOT/scripts/run_groove.sh" "$@"
