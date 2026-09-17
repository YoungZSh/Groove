#!/usr/bin/env bash
# Single-node, four-GPU Qwen3.5-2B GRPO or GRPO + OPSD on Siton.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TRAINING_MODE="${TRAINING_MODE:-grpo}"
case "$TRAINING_MODE" in
  grpo) LAUNCHER="$PROJECT_ROOT/TMP/scripts/run_grpo_2b.sh" ;;
  grpo_opsd) LAUNCHER="$PROJECT_ROOT/TMP/scripts/run_grpo_opsd_2b.sh" ;;
  *) echo "TRAINING_MODE must be grpo or grpo_opsd." >&2; exit 2 ;;
esac
: "${EXPERIMENT_NAME:?Set a new EXPERIMENT_NAME for each four-GPU run.}"

export PYTHON_BIN="${PYTHON_BIN:-/home/yzs/miniconda3/envs/vision-opd/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
if [[ "${N_GPUS:-4}" != "4" ]]; then
  echo "This launcher requires N_GPUS=4." >&2
  exit 2
fi
export N_GPUS=4
export SEED="${SEED:-20260904}"
# Use Ray's node-relative memory threshold on the large-memory four-GPU host.
# An explicit numeric value remains available for a constrained allocation.
export RAY_NODE_MEMORY_CAP_GIB="${RAY_NODE_MEMORY_CAP_GIB:-null}"

# Increase local scheduling capacity and per-GPU packed-token budgets while
# keeping the global prompt batch (16) and rollouts per group (8) unchanged.
export ROLLOUT_AGENT_NUM_WORKERS="${ROLLOUT_AGENT_NUM_WORKERS:-16}"
export REWARD_NUM_WORKERS="${REWARD_NUM_WORKERS:-4}"
export ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU="${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU:-65536}"
export ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-65536}"
# Submit the full validation set together so vLLM can continuously schedule
# requests instead of waiting for every small batch to finish. Hydra null makes
# the validation loader use the dataset length; padding is removed after rollout.
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-null}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

IFS=',' read -r -a selected_gpus <<< "$CUDA_VISIBLE_DEVICES"
if [[ ${#selected_gpus[@]} -ne 4 ]]; then
  echo "CUDA_VISIBLE_DEVICES must select exactly four distinct GPUs." >&2
  exit 2
fi
declare -A seen_gpus=()
for gpu in "${selected_gpus[@]}"; do
  if [[ -z "$gpu" || "$gpu" == *[[:space:]]* || -n "${seen_gpus[$gpu]:-}" ]]; then
    echo "CUDA_VISIBLE_DEVICES must select exactly four distinct GPUs." >&2
    exit 2
  fi
  seen_gpus[$gpu]=1
done
for argument in "$@"; do
  key="${argument%%=*}"
  key="${key//+/}"
  case "$key" in
    trainer.nnodes)
      [[ "${argument#*=}" == "1" ]] || { echo "This launcher is single-node only." >&2; exit 2; } ;;
    trainer.n_gpus_per_node)
      [[ "${argument#*=}" == "4" ]] || { echo "This launcher requires four GPUs per node." >&2; exit 2; } ;;
  esac
done

# The neighbouring GLaQ/mmcot GRPO launchers use loopback + disabled IB/RoCE
# for single-node NCCL bootstrap. Forward both settings to Ray workers too.
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NO_PROXY="127.0.0.1,localhost,::1${NO_PROXY:+,$NO_PROXY}${no_proxy:+,$no_proxy}"
for address in $(hostname -I 2>/dev/null || true); do
  NO_PROXY+=",$address"
done
export no_proxy="$NO_PROXY"
RUNTIME_ENV_OVERRIDES=()
for variable in NCCL_SOCKET_IFNAME NCCL_IB_DISABLE NO_PROXY no_proxy OMP_NUM_THREADS; do
  RUNTIME_ENV_OVERRIDES+=("++ray_kwargs.ray_init.runtime_env.env_vars.$variable=\"${!variable}\"")
done

dry_run="${GROOVE_DRY_RUN:-false}"
case "${dry_run,,}" in
  1|true|yes)
    exec bash "$LAUNCHER" "${RUNTIME_ENV_OVERRIDES[@]}" "$@"
    ;;
esac

# Check the selected CUDA devices before Ray allocates workers or model weights.
"$PYTHON_BIN" - <<'PY'
import os
import torch

count = torch.cuda.device_count()
if count != 4:
    raise SystemExit(
        f"Four GPUs are required, but CUDA can see {count} with "
        f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}. "
        "Expose four GPUs to this container, or use GROOVE_DRY_RUN=true "
        "to validate the configuration only."
    )
PY

LOG_FILE="$PROJECT_ROOT/outputs/logs/$EXPERIMENT_NAME.log"
mkdir -p "$(dirname "$LOG_FILE")"
bash "$LAUNCHER" "${RUNTIME_ENV_OVERRIDES[@]}" "$@" 2>&1 | tee -a "$LOG_FILE"
