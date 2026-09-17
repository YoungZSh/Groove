#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/yzs/miniconda3/envs/vision-opd/bin/python}"
MODEL_PATH="${MODEL_PATH:-/root/siton-tmp/yzs/ckpts/Qwen3.5-4B}"

# This launcher is intentionally scoped to the first 1.1K examples.  Keep the
# generated split separate from the historical 6K artifacts.
SEED="${SEED:-1234}"
DATA_MAX_SOURCE_ROWS="${DATA_MAX_SOURCE_ROWS:-1100}"
TEST_RATIO="${TEST_RATIO:-0.10}"
# Run the requested GRPO-only ablation by default. Set OPSD_ENABLED=true to
# restore the historical joint GRPO+OPSD objective. Disabling OPSD here also
# removes the Analyzer/Teacher dependency instead of merely zeroing its loss.
OPSD_ENABLED="${OPSD_ENABLED:-false}"
case "$OPSD_ENABLED" in
  true|false) ;;
  *)
    echo "OPSD_ENABLED must be true or false, got: $OPSD_ENABLED" >&2
    exit 2
    ;;
esac
if [[ "$OPSD_ENABLED" == "true" ]]; then
  DEFAULT_TRAIN_BATCH_SIZE=32
else
  DEFAULT_TRAIN_BATCH_SIZE=16
fi
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$DEFAULT_TRAIN_BATCH_SIZE}"
ROLLOUT_N="${ROLLOUT_N:-8}"
# VERL 0.9 defines this in prompt groups and multiplies by rollout.n inside
# the trainer before dispatching the completion batch.
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-$TRAIN_BATCH_SIZE}"
DATA_OUTPUT_DIR="${DATA_OUTPUT_DIR:-$PROJECT_ROOT/data/vision_opd_${DATA_MAX_SOURCE_ROWS}_seed${SEED}}"
TRAIN_FILE="${TRAIN_FILE:-$DATA_OUTPUT_DIR/train.parquet}"
TEST_FILE="${TEST_FILE:-$DATA_OUTPUT_DIR/test.parquet}"
if [[ "$OPSD_ENABLED" == "true" ]]; then
  DEFAULT_EXPERIMENT_PREFIX="qwen35-4b-groove-visual-evidence"
else
  DEFAULT_EXPERIMENT_PREFIX="qwen35-4b-grpo"
fi
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${DEFAULT_EXPERIMENT_PREFIX}-${DATA_MAX_SOURCE_ROWS}-seed${SEED}-b${TRAIN_BATCH_SIZE}}"

if [[ "${PREPARE_DATA:-true}" == "true" ]]; then
  "$PYTHON_BIN" "$PROJECT_ROOT/TMP/scripts/prepare_vision_opd.py" \
    --output-dir "$DATA_OUTPUT_DIR" \
    --test-ratio "$TEST_RATIO" \
    --random-state "$SEED" \
    --max-source-rows "$DATA_MAX_SOURCE_ROWS"
elif [[ ! -f "$TRAIN_FILE" || ! -f "$TEST_FILE" ]]; then
  echo "Missing prepared data; required training file: $TRAIN_FILE; validation file: $TEST_FILE." >&2
  echo "Prepare the requested splits first; V*Bench validation uses scripts/prepare_vstar_validation.py." >&2
  exit 2
fi
if [[ "$OPSD_ENABLED" == "true" ]]; then
  if [[ -z "${ANALYZER_BASE_URL:-}" || -z "${ANALYZER_API_KEY:-}" ]]; then
    echo "Set ANALYZER_BASE_URL and ANALYZER_API_KEY before starting OPSD training." >&2
    exit 2
  fi
fi

export VERL_CONFIG_NAME="groove"
export GROOVE_REQUIRE_SLEEP_LEVEL_2="${GROOVE_REQUIRE_SLEEP_LEVEL_2:-true}"
# Both the project package and its vendored ``verl`` runtime live in src.
# Keep this checkout first so an unrelated ``verl`` installation in the Conda
# environment cannot silently take precedence.
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
# Qwen3.5's DeltaNet dependency was built against the conda toolchain.  Keep
# the environment lib directory first so actor workers do not accidentally use
# the host libstdc++ when Ray starts a fresh process.
PYTHON_PREFIX="$(cd "$(dirname "$PYTHON_BIN")/.." && pwd)"
export LD_LIBRARY_PATH="$PYTHON_PREFIX/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export ANALYZER_MODEL="${ANALYZER_MODEL:-gpt-5.6}"
export ANALYZER_TEMPERATURE="${ANALYZER_TEMPERATURE:-0.0}"
export ANALYZER_MAX_COMPLETION_TOKENS="${ANALYZER_MAX_COMPLETION_TOKENS:-512}"
export ANALYZER_API_RETRIES="${ANALYZER_API_RETRIES:-5}"
export ANALYZER_API_RETRY_DELAY="${ANALYZER_API_RETRY_DELAY:-1.0}"
export GROOVE_EVIDENCE_DIR="${GROOVE_EVIDENCE_DIR:-$PROJECT_ROOT/outputs/evidence}"
export GROOVE_MIXED_GROUPS_ONLY="${GROOVE_MIXED_GROUPS_ONLY:-false}"
# The remote Analyzer vLLM service is configured with max-num-seqs=64. Run 16
# evidence groups concurrently; DINO/OCR remain single-worker services behind
# their HTTP endpoints and may queue tool requests without changing worker count.
export GROOVE_MAX_CONCURRENCY="${GROOVE_MAX_CONCURRENCY:-16}"
# Bound each original/crop image in the Teacher prefix independently. This
# prevents one high-resolution multi-crop example from exceeding its text
# context budget while retaining all selected evidence images.
export GROOVE_TEACHER_MAX_IMAGE_PIXELS="${GROOVE_TEACHER_MAX_IMAGE_PIXELS:-1048576}"
export GROUNDING_DINO_MODEL="${GROUNDING_DINO_MODEL:-IDEA-Research/grounding-dino-base}"
export GROUNDING_DINO_DEVICE="${GROUNDING_DINO_DEVICE:-cpu}"
export GROUNDING_DINO_LOCAL_FILES_ONLY="${GROUNDING_DINO_LOCAL_FILES_ONLY:-true}"
export RAY_NODE_MEMORY_CAP_GIB="${RAY_NODE_MEMORY_CAP_GIB:-220}"
export RAY_MEMORY_GUARD_HEADROOM_GIB="${RAY_MEMORY_GUARD_HEADROOM_GIB:-4}"
export RAY_MEMORY_USAGE_THRESHOLD_CEILING="${RAY_MEMORY_USAGE_THRESHOLD_CEILING:-0.95}"
export RAY_OBJECT_STORE_GIB="${RAY_OBJECT_STORE_GIB:-8}"
export RAY_MEMORY_MONITOR_REFRESH_MS="${RAY_MEMORY_MONITOR_REFRESH_MS:-100}"
export PYTHONBUFFERED=1
export VLLM_USE_V1=1
unset VLLM_ATTENTION_BACKEND

MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
ENABLE_THINKING="${ENABLE_THINKING:-false}"
STUDENT_RESPONSE_FORMAT="${STUDENT_RESPONSE_FORMAT:-original}"
STUDENT_IMAGE_MAX_PIXELS="${STUDENT_IMAGE_MAX_PIXELS:-4194304}"
STUDENT_IMAGE_PATCH_SIZE="${STUDENT_IMAGE_PATCH_SIZE:-16}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-512}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-9216}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
TOTAL_STEPS="${TOTAL_STEPS:-null}"
TEST_FREQ="${TEST_FREQ:-10}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-false}"
SAVE_FREQ="${SAVE_FREQ:-10}"
MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-2}"
N_GPUS="${N_GPUS:-2}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$PROJECT_ROOT/checkpoints/$EXPERIMENT_NAME}"
ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-$PROJECT_ROOT/outputs/rollouts/$EXPERIMENT_NAME}"
TRAINER_LOGGER="${TRAINER_LOGGER:-[\"console\"]}"
ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-true}"
ACTOR_OPTIMIZER_OFFLOAD="${ACTOR_OPTIMIZER_OFFLOAD:-true}"
TRAINING_FSDP_STRATEGY="${TRAINING_FSDP_STRATEGY:-fsdp}"
ACTOR_FSDP_OFFLOAD_POLICY="${ACTOR_FSDP_OFFLOAD_POLICY:-false}"
# Qwen3.5's vision stack currently misindexes VERL's activation-offload
# groups when the Teacher contains several independently cropped images.
# Parameter/optimizer CPU offload still provides the required memory sharing.
ACTOR_ACTIVATION_OFFLOAD="${ACTOR_ACTIVATION_OFFLOAD:-false}"
ACTOR_FSDP_USE_TORCH_COMPILE="${ACTOR_FSDP_USE_TORCH_COMPILE:-false}"
REF_PARAM_OFFLOAD="${REF_PARAM_OFFLOAD:-true}"
ROLLOUT_FREE_CACHE_ENGINE="${ROLLOUT_FREE_CACHE_ENGINE:-true}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.6}"
ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-true}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-16}"
ROLLOUT_TENSOR_PARALLEL_SIZE="${ROLLOUT_TENSOR_PARALLEL_SIZE:-1}"
ROLLOUT_AGENT_NUM_WORKERS="${ROLLOUT_AGENT_NUM_WORKERS:-8}"
ACTOR_USE_TORCH_COMPILE="${ACTOR_USE_TORCH_COMPILE:-false}"
USE_FUSED_KERNELS="${USE_FUSED_KERNELS:-true}"
MODEL_USE_REMOVE_PADDING="${MODEL_USE_REMOVE_PADDING:-true}"
FUSED_ADAMW="${FUSED_ADAMW:-true}"
OPTIMIZER_IMPL="${OPTIMIZER_IMPL:-torch.optim}"
OPTIMIZER_NAME="${OPTIMIZER_NAME:-AdamW}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
PPO_CLIP_RATIO="${PPO_CLIP_RATIO:-0.2}"
# Match the historical VERL behavior unless an experiment launcher explicitly
# requests the per-rollout normalization used by the original GRPO objective.
LOSS_AGG_MODE="${LOSS_AGG_MODE:-token-mean}"
REFERENCE_KL_COEF="${REFERENCE_KL_COEF:-0.001}"
OPSD_ADVANTAGE_COEF="${OPSD_ADVANTAGE_COEF:-0.01}"
OPSD_ADVANTAGE_CLIP="${OPSD_ADVANTAGE_CLIP:-null}"
OPSD_LOG_PROB_DUMP_DIR="${OPSD_LOG_PROB_DUMP_DIR:-$PROJECT_ROOT/outputs/opsd-token-dumps/$EXPERIMENT_NAME}"
ANSWER_REWARD_WEIGHT="${ANSWER_REWARD_WEIGHT:-0.9}"
FORMAT_REWARD_WEIGHT="${FORMAT_REWARD_WEIGHT:-0.1}"
CUSTOM_REWARD_FUNCTION_PATH="${CUSTOM_REWARD_FUNCTION_PATH:-$PROJECT_ROOT/src/groove/reward.py}"
CUSTOM_REWARD_FUNCTION_NAME="${CUSTOM_REWARD_FUNCTION_NAME:-compute_score}"
REWARD_MANAGER_NAME="${REWARD_MANAGER_NAME:-naive}"
REWARD_NUM_WORKERS="${REWARD_NUM_WORKERS:-8}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-$MAX_MODEL_LEN}"
ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU="${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU:-$MAX_MODEL_LEN}"

# `fused` and `foreach` are PyTorch AdamW-specific constructor arguments.
# In the constrained one-GPU training topology the launcher selects
# bitsandbytes AdamW8bit, whose optimizer state is compact enough to coexist
# with the actor during the update phase; do not forward PyTorch-only flags.
OPTIMIZER_OVERRIDES=(
  "actor_rollout_ref.actor.optim.optimizer_impl=$OPTIMIZER_IMPL"
  "actor_rollout_ref.actor.optim.optimizer=$OPTIMIZER_NAME"
)
if [[ "$OPTIMIZER_IMPL" == "torch.optim" && "$OPTIMIZER_NAME" == "AdamW" ]]; then
  OPTIMIZER_OVERRIDES+=(
    "actor_rollout_ref.actor.optim.override_optimizer_config.fused=$FUSED_ADAMW"
    "actor_rollout_ref.actor.optim.override_optimizer_config.foreach=false"
  )
else
  OPTIMIZER_OVERRIDES+=("actor_rollout_ref.actor.optim.override_optimizer_config=null")
fi
if [[ "$OPTIMIZER_IMPL" == "bitsandbytes.optim" ]]; then
  "$PYTHON_BIN" -c 'import bitsandbytes' \
    || { echo "bitsandbytes optimizer selected but unavailable in $PYTHON_BIN" >&2; exit 2; }
fi

if [[ "$OPSD_ENABLED" == "true" ]]; then
  OPSD_OVERRIDES=(
    "actor_rollout_ref.actor.policy_loss.loss_mode=vanilla"
    "groove.enabled=true"
    "groove.opsd_advantage_coef=$OPSD_ADVANTAGE_COEF"
    "groove.opsd_advantage_clip=$OPSD_ADVANTAGE_CLIP"
    "groove.max_reprompt_len=$MAX_MODEL_LEN"
  )
else
  # Vanilla policy loss is the actual GRPO-only path. The project-level flag
  # is checked before constructing any privileged Teacher/evidence inputs.
  OPSD_OVERRIDES=(
    "actor_rollout_ref.actor.policy_loss.loss_mode=vanilla"
    "groove.enabled=false"
  )
fi

exec "$PYTHON_BIN" -m groove.verl_entrypoint \
  "data.train_files=['$TRAIN_FILE']" \
  "data.val_files=['$TEST_FILE']" \
  data.val_batch_size="${VAL_BATCH_SIZE:-null}" \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.response_format="$STUDENT_RESPONSE_FORMAT" \
  data.max_prompt_length="$MAX_PROMPT_LENGTH" \
  data.apply_chat_template_kwargs.enable_thinking="$ENABLE_THINKING" \
  data.image_max_pixels="$STUDENT_IMAGE_MAX_PIXELS" \
  data.image_patch_size="$STUDENT_IMAGE_PATCH_SIZE" \
  data.max_response_length="$MAX_RESPONSE_LENGTH" \
  data.filter_overlong_prompts=false \
  data.truncation=error \
  data.shuffle=true \
  data.seed="$SEED" \
  data.return_multi_modal_inputs=true \
  data.dataloader_num_workers=0 \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.trust_remote_code=true \
  actor_rollout_ref.model.use_remove_padding="$MODEL_USE_REMOVE_PADDING" \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.model.enable_activation_offload="$ACTOR_ACTIVATION_OFFLOAD" \
  actor_rollout_ref.model.use_fused_kernels="$USE_FUSED_KERNELS" \
  actor_rollout_ref.model.fused_kernel_options.impl_backend=torch \
  actor_rollout_ref.actor.use_torch_compile="$ACTOR_USE_TORCH_COMPILE" \
  actor_rollout_ref.ref.use_torch_compile=false \
  actor_rollout_ref.actor.fsdp_config.use_torch_compile="$ACTOR_FSDP_USE_TORCH_COMPILE" \
  actor_rollout_ref.ref.fsdp_config.use_torch_compile=false \
  actor_rollout_ref.hybrid_engine=true \
  actor_rollout_ref.rollout.n="$ROLLOUT_N" \
  actor_rollout_ref.rollout.seed="$SEED" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.agent.num_workers="$ROLLOUT_AGENT_NUM_WORKERS" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TENSOR_PARALLEL_SIZE" \
  actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEMORY_UTILIZATION" \
  actor_rollout_ref.rollout.free_cache_engine="$ROLLOUT_FREE_CACHE_ENGINE" \
  actor_rollout_ref.rollout.enable_sleep_mode=true \
  actor_rollout_ref.rollout.layered_summon=false \
  actor_rollout_ref.rollout.enforce_eager="$ROLLOUT_ENFORCE_EAGER" \
  actor_rollout_ref.rollout.load_format=dummy \
  actor_rollout_ref.rollout.enable_chunked_prefill=true \
  actor_rollout_ref.rollout.max_num_seqs="$ROLLOUT_MAX_NUM_SEQS" \
  actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN" \
  actor_rollout_ref.rollout.max_num_batched_tokens="$ROLLOUT_MAX_NUM_BATCHED_TOKENS" \
  actor_rollout_ref.rollout.response_length="$MAX_RESPONSE_LENGTH" \
  actor_rollout_ref.rollout.calculate_log_probs=true \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.optim.lr="$LEARNING_RATE" \
  "${OPTIMIZER_OVERRIDES[@]}" \
  actor_rollout_ref.actor.strategy="$TRAINING_FSDP_STRATEGY" \
  actor_rollout_ref.actor.fsdp_config.strategy="$TRAINING_FSDP_STRATEGY" \
  actor_rollout_ref.actor.fsdp_config.offload_policy="$ACTOR_FSDP_OFFLOAD_POLICY" \
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.data_loader_seed="$SEED" \
  actor_rollout_ref.actor.fsdp_config.seed="$SEED" \
  actor_rollout_ref.actor.clip_ratio="$PPO_CLIP_RATIO" \
  actor_rollout_ref.actor.clip_ratio_low="$PPO_CLIP_RATIO" \
  actor_rollout_ref.actor.clip_ratio_high="$PPO_CLIP_RATIO" \
  actor_rollout_ref.actor.loss_agg_mode="$LOSS_AGG_MODE" \
  actor_rollout_ref.actor.entropy_coeff=0.0 \
  actor_rollout_ref.actor.use_dynamic_bsz=true \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU" \
  actor_rollout_ref.actor.fsdp_config.param_offload="$ACTOR_PARAM_OFFLOAD" \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="$ACTOR_OPTIMIZER_OFFLOAD" \
  actor_rollout_ref.actor.fsdp_config.reshard_after_forward=true \
  actor_rollout_ref.actor.use_kl_loss=true \
  actor_rollout_ref.actor.kl_loss_coef="$REFERENCE_KL_COEF" \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.seed="$SEED" \
  actor_rollout_ref.ref.fsdp_config.param_offload="$REF_PARAM_OFFLOAD" \
  "${OPSD_OVERRIDES[@]}" \
  algorithm.adv_estimator=grpo \
  algorithm.norm_adv_by_std_in_grpo=true \
  algorithm.use_kl_in_reward=false \
  reward.reward_model.enable=false \
  reward.num_workers="$REWARD_NUM_WORKERS" \
  reward.reward_manager.name="$REWARD_MANAGER_NAME" \
  reward.custom_reward_function.path="$CUSTOM_REWARD_FUNCTION_PATH" \
  reward.custom_reward_function.name="$CUSTOM_REWARD_FUNCTION_NAME" \
  reward.custom_reward_function.reward_kwargs.answer_reward_weight="$ANSWER_REWARD_WEIGHT" \
  reward.custom_reward_function.reward_kwargs.format_reward_weight="$FORMAT_REWARD_WEIGHT" \
  trainer.project_name=groove-visual-evidence \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.logger="$TRAINER_LOGGER" \
  trainer.n_gpus_per_node="$N_GPUS" \
  trainer.nnodes=1 \
  trainer.total_epochs="$TOTAL_EPOCHS" \
  trainer.total_training_steps="$TOTAL_STEPS" \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.max_actor_ckpt_to_keep="$MAX_ACTOR_CKPT_TO_KEEP" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.val_before_train="$VAL_BEFORE_TRAIN" \
  trainer.use_v1=false \
  trainer.default_local_dir="$CHECKPOINT_DIR" \
  trainer.rollout_data_dir="$ROLLOUT_DATA_DIR" \
  "$@"
