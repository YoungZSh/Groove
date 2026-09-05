#!/usr/bin/env bash
# GRPO + OPSD Qwen3.5-2B run with ordinary reasoning and a terminal answer.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data/deepeyes_vstar_opsd_2200_seed20260904"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen35-2b-grpo-opsd-reasoning-answer-seed20260904}"

# Keep the established model, split, sampling, and optimizer settings. Both 2B
# launchers now use plain reasoning plus answer tags; their seeds still differ.
export MODEL_PATH="/root/siton-tmp/yzs/ckpts/Qwen3.5-2B"
export PREPARE_DATA=false
export DATA_OUTPUT_DIR="$DATA_DIR"
export TRAIN_FILE="$DATA_DIR/train.parquet"
export TEST_FILE="$DATA_DIR/validation.parquet"
export SEED=20260904

export TRAINER_LOGGER='["console","wandb"]'
export WANDB_MODE=offline
export WANDB_PROJECT=groove-visual-evidence
export WANDB_NAME="$EXPERIMENT_NAME"
export WANDB_DIR="$PROJECT_ROOT/outputs/wandb"

export CUDA_VISIBLE_DEVICES=0,1
export N_GPUS=2
export OPSD_ENABLED=true
export OPSD_ADVANTAGE_COEF=0.01
export OPSD_ADVANTAGE_CLIP=null
export GROOVE_REQUIRE_SLEEP_LEVEL_2=false

export ANALYZER_BASE_URL="${ANALYZER_BASE_URL:-http://127.0.0.1:8002/v1}"
export ANALYZER_API_KEY="${ANALYZER_API_KEY:-unused}"
export ANALYZER_MODEL="${ANALYZER_MODEL:-Qwen3.8-27B}"
export ANALYZER_DISABLE_THINKING=true
export ANALYZER_USE_VISION_TOOLS=true
export ANALYZER_MAX_TOOL_ROUNDS=3
export ANALYZER_MAX_COMPLETION_TOKENS=2048
export ANALYZER_TOOL_FEEDBACK_MAX_SIDE=1024
export ANALYZER_GROUNDING_URL="${ANALYZER_GROUNDING_URL:-http://127.0.0.1:8011}"
export ANALYZER_OCR_URL="${ANALYZER_OCR_URL:-http://127.0.0.1:8012}"
# Only Analyzer group orchestration is widened. The remote DINO and OCR
# services intentionally remain one worker each.
export GROOVE_MAX_CONCURRENCY=16
export GROOVE_MIXED_GROUPS_ONLY=false
export GROOVE_EVIDENCE_DIR="$PROJECT_ROOT/outputs/evidence/$EXPERIMENT_NAME"
export GROOVE_TEACHER_MAX_IMAGE_PIXELS=1048576

export TRAIN_BATCH_SIZE=16
export ROLLOUT_N=8
export PPO_MINI_BATCH_SIZE=16
# Pool valid response tokens across the batch. With the severe-repetition reward
# gate below, long failed trajectories receive proportionally more gradient.
export LOSS_AGG_MODE="${LOSS_AGG_MODE:-token-mean}"
export ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU="${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU:-32768}"
export MAX_PROMPT_LENGTH=2048
export MAX_RESPONSE_LENGTH=1024
export MAX_MODEL_LEN=9216
export ENABLE_THINKING=false
export STUDENT_RESPONSE_FORMAT=reasoning_answer
export STUDENT_IMAGE_MAX_PIXELS=null
export STUDENT_IMAGE_PATCH_SIZE=16
export MODEL_USE_REMOVE_PADDING=true

export ROLLOUT_TENSOR_PARALLEL_SIZE=1
export ROLLOUT_MAX_NUM_SEQS=64
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=32768
export ROLLOUT_GPU_MEMORY_UTILIZATION=0.45
export ROLLOUT_ENFORCE_EAGER=true

export TRAINING_FSDP_STRATEGY=fsdp
export ACTOR_PARAM_OFFLOAD=false
export ACTOR_OPTIMIZER_OFFLOAD=false
export REF_PARAM_OFFLOAD=false
export ACTOR_ACTIVATION_OFFLOAD=false
export ACTOR_FSDP_OFFLOAD_POLICY=false
export OPTIMIZER_IMPL=torch.optim
export OPTIMIZER_NAME=AdamW
export FUSED_ADAMW=true
export LEARNING_RATE=1e-6
export REFERENCE_KL_COEF=0.01

export CUSTOM_REWARD_FUNCTION_PATH="$PROJECT_ROOT/src/groove/deepeyes_reward.py"
export CUSTOM_REWARD_FUNCTION_NAME=compute_score
export REWARD_MANAGER_NAME=naive
export REWARD_NUM_WORKERS=1
# DeepEyes-style negative-only format penalty while retaining 1.0 as the
# maximum semantic reward: score = accuracy - 0.2 * format_error.
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
export OPSD_LOG_PROB_DUMP_DIR="$PROJECT_ROOT/outputs/opsd-token-dumps/$EXPERIMENT_NAME"

export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
export PYTHONUNBUFFERED=1

exec "$PROJECT_ROOT/scripts/run_groove.sh" "$@"
