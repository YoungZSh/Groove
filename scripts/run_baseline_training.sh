#!/usr/bin/env bash
# Original-image + crop + focus Teacher, with full-trajectory audit records.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHON_BIN="${PYTHON_BIN:-/data/home/yangzesheng/.conda/envs/groove/bin/python}"
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export MODEL_PATH="${MODEL_PATH:-/data/home/yangzesheng/models/ckpts/Qwen3.5-2B}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen35-2b-baseline-1100-seed20260904}"
export BASELINE_RUN_DIR="${BASELINE_RUN_DIR:-$PROJECT_ROOT/outputs/$EXPERIMENT_NAME}"
export PREPARE_DATA=false
export DATA_OUTPUT_DIR="$PROJECT_ROOT/data/baseline_1100_seed20260904"
export TRAIN_FILE="$DATA_OUTPUT_DIR/train.parquet"
export TEST_FILE="$DATA_OUTPUT_DIR/test.parquet"
export SEED=20260904

export ANALYZER_BASE_URL="${ANALYZER_BASE_URL:-http://127.0.0.1:8002/v1}"
export ANALYZER_API_KEY="${ANALYZER_API_KEY:-unused}"
export ANALYZER_MODEL=Qwen3.8-27B
export ANALYZER_GROUNDING_URL="${ANALYZER_GROUNDING_URL:-http://127.0.0.1:8011}"
export ANALYZER_OCR_URL="${ANALYZER_OCR_URL:-http://127.0.0.1:8012}"
export ANALYZER_DISABLE_THINKING=true
export ANALYZER_USE_VISION_TOOLS=true
export ANALYZER_MAX_TOOL_ROUNDS=3
export ANALYZER_MAX_COMPLETION_TOKENS=2048
export ANALYZER_TOOL_FEEDBACK_MAX_SIDE=1024
export GROOVE_MAX_CONCURRENCY=8
export GROOVE_MIXED_GROUPS_ONLY=false
export GROOVE_EVIDENCE_DIR="$BASELINE_RUN_DIR/evidence"
export GROOVE_TEACHER_MAX_IMAGE_PIXELS=1048576
export GROUNDING_DINO_DEVICE=cpu

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export N_GPUS="${N_GPUS:-4}"
export OPSD_ENABLED=true
export OPSD_ADVANTAGE_COEF=0.01
export OPSD_ADVANTAGE_CLIP=null
export GROOVE_REQUIRE_SLEEP_LEVEL_2=false
export TRAIN_BATCH_SIZE=8
export ROLLOUT_N=8
export PPO_MINI_BATCH_SIZE=8
export MAX_PROMPT_LENGTH=8192
export MAX_RESPONSE_LENGTH=1024
export MAX_MODEL_LEN=9216
export STUDENT_IMAGE_MAX_PIXELS=4194304
export STUDENT_IMAGE_PATCH_SIZE=16
export ENABLE_THINKING=false
export MODEL_USE_REMOVE_PADDING=true
export ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU=16384
export ROLLOUT_TENSOR_PARALLEL_SIZE=1
export ROLLOUT_MAX_NUM_SEQS=32
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=16384
export ROLLOUT_GPU_MEMORY_UTILIZATION=0.40
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
export ANSWER_REWARD_WEIGHT=0.9
export FORMAT_REWARD_WEIGHT=0.1
export REWARD_NUM_WORKERS=1
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
export TOTAL_STEPS="${TOTAL_STEPS:-null}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-true}"
export TEST_FREQ="${TEST_FREQ:-10}"
export SAVE_FREQ="${SAVE_FREQ:-10}"
export MAX_ACTOR_CKPT_TO_KEEP=2
export CHECKPOINT_DIR="$BASELINE_RUN_DIR/checkpoints"
export ROLLOUT_DATA_DIR="$BASELINE_RUN_DIR/rollouts"
export OPSD_LOG_PROB_DUMP_DIR="$BASELINE_RUN_DIR/token_audits"
export OMP_NUM_THREADS=8
export NCCL_CUMEM_ENABLE=0
export NCCL_CUMEM_HOST_ENABLE=0
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
export PYTHONUNBUFFERED=1
export GROOVE_WORKER_DEBUG_DIR="$BASELINE_RUN_DIR/worker_debug"

mkdir -p "$BASELINE_RUN_DIR"
exec bash "$PROJECT_ROOT/scripts/run_groove.sh" \
  +ray_kwargs.ray_init.runtime_env.worker_process_setup_hook=groove.runtime_debug.setup_worker_diagnostics \
  ray_kwargs.ray_init.num_cpus=32 \
  "trainer.validation_data_dir=$BASELINE_RUN_DIR/validation" \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=false \
  "$@"
