#!/usr/bin/env bash
# Standalone Qwen3.5-2B experiment on Siton with two GPUs.
# This file is complete: it never sources or calls another launcher.
# Select TRAINING_MODE=grpo, dapo, or grpo_opsd (groove is an alias).
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# ---- Machine settings: edit these when copying this file to a new experiment. ----
export PYTHON_BIN="${PYTHON_BIN:-/home/yzs/miniconda3/envs/vision-opd/bin/python}"
MODEL_PATH="${MODEL_PATH:-/root/siton-tmp/yzs/ckpts/Qwen3.5-2B}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export N_GPUS="${N_GPUS:-2}"
EXPECTED_GPUS=2
export RAY_NODE_MEMORY_CAP_GIB="${RAY_NODE_MEMORY_CAP_GIB:-220}"
ROLLOUT_AGENT_NUM_WORKERS="${ROLLOUT_AGENT_NUM_WORKERS:-8}"
REWARD_NUM_WORKERS="${REWARD_NUM_WORKERS:-1}"
DAPO_MAX_INFLIGHT_GEN_BATCHES="${DAPO_MAX_INFLIGHT_GEN_BATCHES:-1}"
ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU="${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU:-32768}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-32768}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

# ---- Experiment and datasets. Each formal run must have its own name. ----
TRAINING_MODE="${TRAINING_MODE:-grpo}"
: "${EXPERIMENT_NAME:?Set a new EXPERIMENT_NAME for each experiment.}"
SEED="${SEED:-20260904}"
case "$TRAINING_MODE" in
  grpo)
    DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data/vstar_grpo_4000_seed20260917}"
    OPSD_ENABLED=false; USE_VERL_V1=true; DYNAMIC_SAMPLING=false
    USE_REFERENCE_KL=true; REFERENCE_KL_COEF=0.01; PPO_CLIP_RATIO_HIGH=0.2
    export WANDB_MODE="${WANDB_MODE:-online}"
    ;;
  dapo)
    DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data/vstar_grpo_4000_seed20260917}"
    OPSD_ENABLED=false; USE_VERL_V1=true; DYNAMIC_SAMPLING=true
    USE_REFERENCE_KL=false; REFERENCE_KL_COEF=0.0; PPO_CLIP_RATIO_HIGH=0.28
    export WANDB_MODE="${WANDB_MODE:-online}"
    ;;
  grpo_opsd|groove)
    TRAINING_MODE=grpo_opsd
    DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data/vstar_opsd_4000_seed20260917}"
    OPSD_ENABLED=true; USE_VERL_V1=false; DYNAMIC_SAMPLING=false
    USE_REFERENCE_KL=true; REFERENCE_KL_COEF=0.01; PPO_CLIP_RATIO_HIGH=0.2
    export WANDB_MODE=offline
    ;;
  *) echo "TRAINING_MODE must be grpo, dapo, or grpo_opsd (groove)." >&2; exit 2 ;;
esac
TRAIN_FILE="$DATA_DIR/train.parquet"
TEST_FILE="${VALIDATION_FILE:-$PROJECT_ROOT/data/vstar_bench/validation.parquet}"
CHECKPOINT_DIR="$PROJECT_ROOT/checkpoints/$EXPERIMENT_NAME"
ROLLOUT_DATA_DIR="$PROJECT_ROOT/outputs/rollouts/$EXPERIMENT_NAME"
VALIDATION_DATA_DIR="${VALIDATION_DATA_DIR:-$PROJECT_ROOT/outputs/validation/$EXPERIMENT_NAME}"
STEP_TIMING_DIR="${STEP_TIMING_DIR:-$PROJECT_ROOT/outputs/timing/$EXPERIMENT_NAME}"
RESUME_MODE="${RESUME_MODE:-disable}"
RESUME_FROM_PATH="${RESUME_FROM_PATH:-null}"

# ---- Optimization, sampling and evaluation: global prompt groups, not per GPU. ----
TRAIN_BATCH_SIZE=16
PPO_MINI_BATCH_SIZE=16
ROLLOUT_N=8
LEARNING_RATE=1e-6
PPO_CLIP_RATIO=0.2
LOSS_AGG_MODE=token-mean
MAX_PROMPT_LENGTH=9216
MAX_RESPONSE_LENGTH=1024
MAX_MODEL_LEN=10240
ENABLE_THINKING=false
STUDENT_RESPONSE_FORMAT=reasoning_answer
STUDENT_IMAGE_MAX_PIXELS=null
STUDENT_IMAGE_PATCH_SIZE=16
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
TOTAL_STEPS="${TOTAL_STEPS:-null}"
TEST_FREQ="${TEST_FREQ:-10}"
SAVE_FREQ="${SAVE_FREQ:-10}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-true}"
MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-2}"
SAVE_BEST_CHECKPOINT="${SAVE_BEST_CHECKPOINT:-true}"
BEST_CHECKPOINT_METRIC="${BEST_CHECKPOINT_METRIC:-val-core/vstar_bench/reward/mean@1}"

# ---- FSDP and one TP=1 vLLM rollout replica per GPU. ----
TRAINING_FSDP_STRATEGY=fsdp
ACTOR_PARAM_OFFLOAD=false
ACTOR_OPTIMIZER_OFFLOAD=false
ACTOR_FSDP_OFFLOAD_POLICY=false
ACTOR_ACTIVATION_OFFLOAD=false
REF_PARAM_OFFLOAD=false
ACTOR_FSDP_USE_TORCH_COMPILE=false
ACTOR_USE_TORCH_COMPILE=false
USE_FUSED_KERNELS=true
MODEL_USE_REMOVE_PADDING=true
ROLLOUT_TENSOR_PARALLEL_SIZE=1
ROLLOUT_GPU_MEMORY_UTILIZATION=0.45
ROLLOUT_MAX_NUM_SEQS=64
ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-true}"
ROLLOUT_FREE_CACHE_ENGINE=true

# ---- Shared semantic Judge. Validation keeps raw accuracy without reward shaping. ----
CUSTOM_REWARD_FUNCTION_PATH="$PROJECT_ROOT/src/groove/semantic_reward.py"
CUSTOM_REWARD_FUNCTION_NAME=compute_score
ANSWER_REWARD_WEIGHT=1.0
FORMAT_REWARD_WEIGHT=0.2
export GROOVE_JUDGE_BASE_URL="${GROOVE_JUDGE_BASE_URL:-http://127.0.0.1:8002/v1}"
export GROOVE_JUDGE_API_KEY="${GROOVE_JUDGE_API_KEY:-remote-qwen38}"
export GROOVE_JUDGE_MODEL=Qwen3.8-27B
export GROOVE_JUDGE_CONCURRENCY=128
export GROOVE_JUDGE_TIMEOUT_SECONDS=180
export GROOVE_JUDGE_MAX_RETRIES=5
export GROOVE_REPETITION_ZERO_REWARD="${GROOVE_REPETITION_ZERO_REWARD:-true}"
export GROOVE_REPETITION_MIN_REPEATS="${GROOVE_REPETITION_MIN_REPEATS:-4}"
export GROOVE_REPETITION_MIN_TOTAL_CHARACTERS="${GROOVE_REPETITION_MIN_TOTAL_CHARACTERS:-80}"
export GROOVE_REPETITION_MIN_PERIOD="${GROOVE_REPETITION_MIN_PERIOD:-1}"
export GROOVE_REPETITION_MAX_PERIOD="${GROOVE_REPETITION_MAX_PERIOD:-1024}"
export GROOVE_REPETITION_SAMPLE_LENGTH="${GROOVE_REPETITION_SAMPLE_LENGTH:-16}"
export GROOVE_REPETITION_SAMPLE_INTERVAL="${GROOVE_REPETITION_SAMPLE_INTERVAL:-32}"

# ---- GRPO + OPSD only: training-only visual evidence and credit allocation. ----
OPSD_ADVANTAGE_MODE="${OPSD_ADVANTAGE_MODE:-rlsd_positive}"
TEACHER_EVIDENCE_MODE="${TEACHER_EVIDENCE_MODE:-focus}"
FOCUS_BLUR_ALPHA="${FOCUS_BLUR_ALPHA:-0.5}"
FOCUS_BLUR_RADIUS="${FOCUS_BLUR_RADIUS:-12.0}"
RLSD_LAMBDA_INITIAL="${RLSD_LAMBDA_INITIAL:-0.5}"
RLSD_LAMBDA_DECAY_STEPS="${RLSD_LAMBDA_DECAY_STEPS:-50}"
RLSD_CLIP_RANGE="${RLSD_CLIP_RANGE:-0.2}"
RLSD_TEACHER_SYNC_INTERVAL="${RLSD_TEACHER_SYNC_INTERVAL:-10}"
export ANALYZER_BASE_URL="${ANALYZER_BASE_URL:-http://127.0.0.1:8002/v1}"
export ANALYZER_API_KEY="${ANALYZER_API_KEY:-unused}"
export ANALYZER_MODEL=Qwen3.8-27B
export ANALYZER_TEMPERATURE=0.0
export ANALYZER_DISABLE_THINKING=true
export ANALYZER_USE_VISION_TOOLS=true
export ANALYZER_MAX_TOOL_ROUNDS=3
export ANALYZER_MAX_COMPLETION_TOKENS=2048
export ANALYZER_TOOL_FEEDBACK_MAX_SIDE=1024
export ANALYZER_API_RETRIES=5
export ANALYZER_API_RETRY_DELAY=1.0
export ANALYZER_GROUNDING_URL="${ANALYZER_GROUNDING_URL:-http://127.0.0.1:8011}"
export ANALYZER_OCR_URL="${ANALYZER_OCR_URL:-http://127.0.0.1:8012}"
export GROOVE_MAX_CONCURRENCY=16
export GROOVE_MIXED_GROUPS_ONLY=false
export GROOVE_EVIDENCE_DIR="$PROJECT_ROOT/outputs/evidence/$EXPERIMENT_NAME"
export GROOVE_TEACHER_MAX_IMAGE_PIXELS=1048576
export OPSD_LOG_PROB_DUMP_DIR="$PROJECT_ROOT/outputs/opsd-token-dumps/$EXPERIMENT_NAME"
export GROUNDING_DINO_MODEL=IDEA-Research/grounding-dino-base
export GROUNDING_DINO_DEVICE=cpu
export GROUNDING_DINO_LOCAL_FILES_ONLY=true

# ---- Logging and process environment. These settings also reach Ray workers. ----
TRAINER_LOGGER='["console","wandb"]'
export WANDB_PROJECT=groove-visual-evidence
export WANDB_NAME="$EXPERIMENT_NAME"
export WANDB_DIR="$PROJECT_ROOT/outputs/wandb"
export VERL_CONFIG_NAME=groove
export GROOVE_REQUIRE_SLEEP_LEVEL_2=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_PREFIX="$(cd "$(dirname "$PYTHON_BIN")/.." && pwd)"
export LD_LIBRARY_PATH="$PYTHON_PREFIX/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export RAY_MEMORY_GUARD_HEADROOM_GIB="${RAY_MEMORY_GUARD_HEADROOM_GIB:-4}"
export RAY_MEMORY_USAGE_THRESHOLD_CEILING="${RAY_MEMORY_USAGE_THRESHOLD_CEILING:-0.95}"
export RAY_OBJECT_STORE_GIB="${RAY_OBJECT_STORE_GIB:-8}"
export RAY_MEMORY_MONITOR_REFRESH_MS="${RAY_MEMORY_MONITOR_REFRESH_MS:-100}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NO_PROXY="127.0.0.1,localhost,::1${NO_PROXY:+,$NO_PROXY}${no_proxy:+,$no_proxy}"
for address in $(hostname -I 2>/dev/null || true); do NO_PROXY+=",$address"; done
export no_proxy="$NO_PROXY"
unset VLLM_USE_V1 VLLM_ATTENTION_BACKEND

# ---- Fail before allocating GPU workers if the experiment is inconsistent. ----
if [[ "$N_GPUS" != "$EXPECTED_GPUS" ]]; then
  echo "This experiment requires N_GPUS=$EXPECTED_GPUS; edit its machine block for a different topology." >&2; exit 2
fi
IFS=',' read -r -a selected_gpus <<< "$CUDA_VISIBLE_DEVICES"
if [[ ${#selected_gpus[@]} -ne $EXPECTED_GPUS ]]; then
  echo "CUDA_VISIBLE_DEVICES must select exactly $EXPECTED_GPUS distinct GPUs." >&2; exit 2
fi
declare -A seen_gpus=()
for gpu in "${selected_gpus[@]}"; do
  if [[ -z "$gpu" || "$gpu" == *[[:space:]]* || -n "${seen_gpus[$gpu]:-}" ]]; then
    echo "CUDA_VISIBLE_DEVICES must select distinct nonempty GPU IDs." >&2; exit 2
  fi
  seen_gpus[$gpu]=1
done
for argument in "$@"; do
  key="${argument%%=*}"; key="${key//+/}"
  case "$key" in
    trainer.nnodes) [[ "${argument#*=}" == "1" ]] || { echo "Single-node experiment required." >&2; exit 2; } ;;
    trainer.n_gpus_per_node) [[ "${argument#*=}" == "$EXPECTED_GPUS" ]] || { echo "GPU count differs from machine settings." >&2; exit 2; } ;;
    *enable_thinking) [[ "${argument#*=}" == "false" ]] || { echo "Thinking must remain disabled." >&2; exit 2; } ;;
  esac
done
if [[ ! -f "$TRAIN_FILE" || ! -f "$TEST_FILE" ]]; then
  echo "Missing prepared data: $TRAIN_FILE or $TEST_FILE" >&2; exit 2
fi

########################### parameter arrays ###########################

# Training and validation data; Student input format.
DATA=(
  "data.train_files=['$TRAIN_FILE']"
  "data.val_files=['$TEST_FILE']"
  data.val_batch_size="${VAL_BATCH_SIZE:-null}"
  data.train_batch_size="$TRAIN_BATCH_SIZE"
  data.response_format="$STUDENT_RESPONSE_FORMAT"
  data.max_prompt_length="$MAX_PROMPT_LENGTH"
  data.apply_chat_template_kwargs.enable_thinking="$ENABLE_THINKING"
  data.image_max_pixels="$STUDENT_IMAGE_MAX_PIXELS"
  data.image_patch_size="$STUDENT_IMAGE_PATCH_SIZE"
  data.max_response_length="$MAX_RESPONSE_LENGTH"
  data.filter_overlong_prompts=false
  data.truncation=error
  data.shuffle=true
  data.seed="$SEED"
  data.return_multi_modal_inputs=true
  data.dataloader_num_workers=0
)

# Shared model and hybrid-engine settings.
MODEL=(
  actor_rollout_ref.model.path="$MODEL_PATH"
  actor_rollout_ref.model.trust_remote_code=true
  actor_rollout_ref.model.use_remove_padding="$MODEL_USE_REMOVE_PADDING"
  actor_rollout_ref.model.enable_gradient_checkpointing=true
  actor_rollout_ref.model.enable_activation_offload="$ACTOR_ACTIVATION_OFFLOAD"
  actor_rollout_ref.model.use_fused_kernels="$USE_FUSED_KERNELS"
  actor_rollout_ref.model.fused_kernel_options.impl_backend=torch
  actor_rollout_ref.hybrid_engine=true
)

# Optimizer, PPO loss and FSDP updates.
ACTOR=(
  actor_rollout_ref.actor.use_torch_compile="$ACTOR_USE_TORCH_COMPILE"
  actor_rollout_ref.actor.fsdp_config.use_torch_compile="$ACTOR_FSDP_USE_TORCH_COMPILE"
  actor_rollout_ref.actor.optim.lr="$LEARNING_RATE"
  actor_rollout_ref.actor.optim.optimizer_impl=torch.optim
  actor_rollout_ref.actor.optim.optimizer=AdamW
  actor_rollout_ref.actor.optim.override_optimizer_config.fused=true
  actor_rollout_ref.actor.optim.override_optimizer_config.foreach=false
  actor_rollout_ref.actor.strategy="$TRAINING_FSDP_STRATEGY"
  actor_rollout_ref.actor.fsdp_config.strategy="$TRAINING_FSDP_STRATEGY"
  actor_rollout_ref.actor.fsdp_config.offload_policy="$ACTOR_FSDP_OFFLOAD_POLICY"
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE"
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
  actor_rollout_ref.actor.ppo_epochs=1
  actor_rollout_ref.actor.data_loader_seed="$SEED"
  actor_rollout_ref.actor.fsdp_config.seed="$SEED"
  actor_rollout_ref.actor.clip_ratio="$PPO_CLIP_RATIO"
  actor_rollout_ref.actor.clip_ratio_low="$PPO_CLIP_RATIO"
  actor_rollout_ref.actor.clip_ratio_high="$PPO_CLIP_RATIO_HIGH"
  actor_rollout_ref.actor.loss_agg_mode="$LOSS_AGG_MODE"
  actor_rollout_ref.actor.entropy_coeff=0.0
  actor_rollout_ref.actor.use_dynamic_bsz=true
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU"
  actor_rollout_ref.actor.fsdp_config.param_offload="$ACTOR_PARAM_OFFLOAD"
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="$ACTOR_OPTIMIZER_OFFLOAD"
  actor_rollout_ref.actor.fsdp_config.reshard_after_forward=true
  actor_rollout_ref.actor.use_kl_loss="$USE_REFERENCE_KL"
  actor_rollout_ref.actor.kl_loss_coef="$REFERENCE_KL_COEF"
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  actor_rollout_ref.actor.policy_loss.loss_mode=vanilla
)

# vLLM sampling and validation generation.
ROLLOUT=(
  actor_rollout_ref.rollout.n="$ROLLOUT_N"
  actor_rollout_ref.rollout.temperature=1.0
  actor_rollout_ref.rollout.top_p=1.0
  actor_rollout_ref.rollout.top_k=-1
  actor_rollout_ref.rollout.val_kwargs.temperature=0.0
  actor_rollout_ref.rollout.val_kwargs.do_sample=false
  actor_rollout_ref.rollout.val_kwargs.n=1
  actor_rollout_ref.rollout.seed="$SEED"
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.mode=async
  actor_rollout_ref.rollout.agent.num_workers="$ROLLOUT_AGENT_NUM_WORKERS"
  actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TENSOR_PARALLEL_SIZE"
  actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEMORY_UTILIZATION"
  actor_rollout_ref.rollout.free_cache_engine="$ROLLOUT_FREE_CACHE_ENGINE"
  actor_rollout_ref.rollout.enable_sleep_mode=true
  actor_rollout_ref.rollout.layered_summon=false
  actor_rollout_ref.rollout.enforce_eager="$ROLLOUT_ENFORCE_EAGER"
  actor_rollout_ref.rollout.load_format=dummy
  actor_rollout_ref.rollout.enable_chunked_prefill=true
  actor_rollout_ref.rollout.max_num_seqs="$ROLLOUT_MAX_NUM_SEQS"
  actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN"
  actor_rollout_ref.rollout.max_num_batched_tokens="$ROLLOUT_MAX_NUM_BATCHED_TOKENS"
  actor_rollout_ref.rollout.response_length="$MAX_RESPONSE_LENGTH"
  actor_rollout_ref.rollout.calculate_log_probs=true
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
)

# Frozen reference-policy scoring.
REF=(
  actor_rollout_ref.ref.use_torch_compile=false
  actor_rollout_ref.ref.fsdp_config.use_torch_compile=false
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
  actor_rollout_ref.ref.fsdp_config.seed="$SEED"
  actor_rollout_ref.ref.fsdp_config.param_offload="$REF_PARAM_OFFLOAD"
)

# GRPO advantages and optional DAPO group filtering.
ALGORITHM=(
  algorithm.adv_estimator=grpo
  algorithm.norm_adv_by_std_in_grpo=true
  algorithm.use_kl_in_reward=false
  algorithm.filter_groups.enable="$DYNAMIC_SAMPLING"
  algorithm.filter_groups.metric=training_reward
  algorithm.filter_groups.max_num_gen_batches=0
  algorithm.filter_groups.max_inflight_gen_batches="$DAPO_MAX_INFLIGHT_GEN_BATCHES"
)

# Semantic Judge adapter and training-only reward shaping.
REWARD=(
  reward.reward_model.enable=false
  reward.num_workers="$REWARD_NUM_WORKERS"
  reward.reward_manager.source=importlib
  reward.reward_manager.name=VisualQARewardManager
  reward.reward_manager.module.path="$PROJECT_ROOT/src/groove/reward_manager.py"
  # The generation limit does not change the reward in any training mode.
  ++reward.reward_kwargs.overlong_buffer_cfg.enable=false
  reward.custom_reward_function.path="$CUSTOM_REWARD_FUNCTION_PATH"
  reward.custom_reward_function.name="$CUSTOM_REWARD_FUNCTION_NAME"
  reward.custom_reward_function.reward_kwargs.answer_reward_weight="$ANSWER_REWARD_WEIGHT"
  reward.custom_reward_function.reward_kwargs.format_reward_weight="$FORMAT_REWARD_WEIGHT"
)

# Optional visual Teacher credit allocation (legacy additive OPSD remains selectable).
OPSD=(
  "groove.enabled=$OPSD_ENABLED"
  "groove.advantage_mode=$OPSD_ADVANTAGE_MODE"
  "groove.teacher_evidence_mode=$TEACHER_EVIDENCE_MODE"
  "groove.focus_blur_alpha=$FOCUS_BLUR_ALPHA"
  "groove.focus_blur_radius=$FOCUS_BLUR_RADIUS"
  "groove.rlsd_lambda_initial=$RLSD_LAMBDA_INITIAL"
  "groove.rlsd_lambda_decay_steps=$RLSD_LAMBDA_DECAY_STEPS"
  "groove.rlsd_clip_range=$RLSD_CLIP_RANGE"
  "groove.rlsd_teacher_sync_interval=$RLSD_TEACHER_SYNC_INTERVAL"
  groove.opsd_advantage_coef=0.01
  groove.opsd_advantage_clip=null
  "groove.max_reprompt_len=$MAX_MODEL_LEN"
)

# Experiment identity, logging, checkpoints and execution backend.
TRAINER=(
  trainer.project_name=groove-visual-evidence
  trainer.experiment_name="$EXPERIMENT_NAME"
  trainer.logger="$TRAINER_LOGGER"
  trainer.step_timing_dir="$STEP_TIMING_DIR"
  trainer.n_gpus_per_node="$N_GPUS"
  trainer.nnodes=1
  trainer.total_epochs="$TOTAL_EPOCHS"
  trainer.total_training_steps="$TOTAL_STEPS"
  trainer.save_freq="$SAVE_FREQ"
  trainer.max_actor_ckpt_to_keep="$MAX_ACTOR_CKPT_TO_KEEP"
  trainer.best_checkpoint.enabled="$SAVE_BEST_CHECKPOINT"
  trainer.best_checkpoint.metric="$BEST_CHECKPOINT_METRIC"
  trainer.best_checkpoint.mode=max
  trainer.test_freq="$TEST_FREQ"
  trainer.val_before_train="$VAL_BEFORE_TRAIN"
  trainer.use_v1="$USE_VERL_V1"
  trainer.v1.trainer_mode=sync
  trainer.resume_mode="$RESUME_MODE"
  trainer.resume_from_path="$RESUME_FROM_PATH"
  trainer.default_local_dir="$CHECKPOINT_DIR"
  trainer.rollout_data_dir="$ROLLOUT_DATA_DIR"
  trainer.validation_data_dir="$VALIDATION_DATA_DIR"
)

# TransferQueue resources and environment forwarded to Ray workers.
RAY=(
  transfer_queue.backend.SimpleStorage.num_data_storage_units=2
)

for variable in NCCL_SOCKET_IFNAME NCCL_IB_DISABLE NO_PROXY no_proxy OMP_NUM_THREADS; do
  RAY+=("++ray_kwargs.ray_init.runtime_env.env_vars.$variable=\"${!variable}\"")
done

# Optional experiment-specific Hydra overrides. CLI arguments take precedence.
EXTRA=()

############################### launch ################################

LAUNCH=("$PYTHON_BIN")
COMMAND=(
  "${LAUNCH[@]}" -m groove.verl_entrypoint
  "${DATA[@]}"
  "${MODEL[@]}"
  "${ACTOR[@]}"
  "${ROLLOUT[@]}"
  "${REF[@]}"
  "${ALGORITHM[@]}"
  "${REWARD[@]}"
  "${OPSD[@]}"
  "${TRAINER[@]}"
  "${RAY[@]}"
  "${EXTRA[@]}"
  "$@"
)

case "${GROOVE_DRY_RUN:-false}" in
  1|true|TRUE|yes) exec "${COMMAND[@]}" ;;
esac
if [[ "$RESUME_MODE" == "disable" && -e "$CHECKPOINT_DIR/latest_checkpointed_iteration.txt" ]]; then
  echo "Checkpoint directory already contains a run; use a new EXPERIMENT_NAME or explicit resume settings." >&2; exit 2
fi
"$PYTHON_BIN" - <<'PY_GPU_CHECK'
import os
import torch
count = torch.cuda.device_count()
expected = int(os.environ["N_GPUS"])
if count != expected:
    raise SystemExit(f"Expected {expected} selected CUDA GPUs, found {count}. Use GROOVE_DRY_RUN=true for configuration checks.")
PY_GPU_CHECK
LOG_FILE="$PROJECT_ROOT/outputs/logs/$EXPERIMENT_NAME.log"
mkdir -p "$(dirname "$LOG_FILE")"
"${COMMAND[@]}" 2>&1 | tee -a "$LOG_FILE"
