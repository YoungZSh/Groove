#!/usr/bin/env bash
# Run a fixed pre-update diagnostic batch through the remote evidence services.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-/data/home/yangzesheng/.conda/envs/groove/bin/python}"
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
MODEL_PATH="${MODEL_PATH:-/data/home/yangzesheng/models/ckpts/Qwen3.5-2B}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/vstar_advantage_10_20260904}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export ANALYZER_BASE_URL="${ANALYZER_BASE_URL:-http://127.0.0.1:8002/v1}"
export ANALYZER_API_KEY="${ANALYZER_API_KEY:-unused}"
export ANALYZER_MODEL="${ANALYZER_MODEL:-Qwen3.8-27B}"
export ANALYZER_GROUNDING_URL="${ANALYZER_GROUNDING_URL:-http://127.0.0.1:8011}"
export ANALYZER_OCR_URL="${ANALYZER_OCR_URL:-http://127.0.0.1:8012}"
export ANALYZER_DISABLE_THINKING=true
export ANALYZER_USE_VISION_TOOLS=true
export ANALYZER_MAX_TOOL_ROUNDS=3
export ANALYZER_MAX_COMPLETION_TOKENS=2048
export ANALYZER_TOOL_FEEDBACK_MAX_SIDE=1024
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
export NCCL_CUMEM_ENABLE=0
export NCCL_CUMEM_HOST_ENABLE=0
export VLLM_ALLREDUCE_USE_SYMM_MEM=0

mkdir -p "$OUTPUT_DIR"
if [[ $# -eq 0 ]]; then
    set -- select rollout evidence score report
fi
for stage in "$@"; do
    # An explicit stage argument supports resuming after a service interruption.
    "$PYTHON_BIN" -u scripts/probe_vstar_advantages.py \
        --stage "$stage" --model "$MODEL_PATH" --output "$OUTPUT_DIR" \
        2>&1 | tee -a "$OUTPUT_DIR/$stage.log"
done
