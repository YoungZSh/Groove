#!/usr/bin/env bash
# Serve the local FP8 Qwen3.5 35B-A3B vision model as the group Analyzer.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/yzs/miniconda3/envs/vision-opd/bin/python}"
MODEL_PATH="${ANALYZER_MODEL_PATH:-/root/siton-tmp/yzs/ckpts/Qwen3.5-35B-A3B-FP8}"
MODEL_NAME="${ANALYZER_MODEL_NAME:-qwen35-35b-a3b-fp8-analyzer}"
GPU_ID="${ANALYZER_GPU_ID:-0}"
HOST="${ANALYZER_HOST:-127.0.0.1}"
PORT="${ANALYZER_PORT:-8001}"
GPU_MEMORY_UTILIZATION="${ANALYZER_GPU_MEMORY_UTILIZATION:-0.74}"
MAX_MODEL_LEN="${ANALYZER_MAX_MODEL_LEN:-12288}"
MAX_NUM_SEQS="${ANALYZER_MAX_NUM_SEQS:-2}"
ENABLE_TOOLS="${ANALYZER_ENABLE_TOOLS:-true}"

checkpoint_ready() {
  "$PYTHON_BIN" - "$MODEL_PATH" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
index = root / "model.safetensors.index.json"
if not (root / "config.json").is_file() or not index.is_file():
    raise SystemExit(1)
try:
    names = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
except (KeyError, OSError, ValueError, TypeError):
    raise SystemExit(1)
raise SystemExit(0 if names and all((root / name).is_file() and (root / name).stat().st_size > 0 for name in names) else 1)
PY
}

if ! checkpoint_ready; then
  "$PYTHON_BIN" "$PROJECT_ROOT/scripts/download_qwen35_analyzer.py" \
    --output-dir "$MODEL_PATH"
fi

# The key only protects a loopback endpoint.  It is intentionally overridable so
# a caller can substitute a stronger local-secret management mechanism.
export CUDA_VISIBLE_DEVICES="$GPU_ID"
PYTHON_PREFIX="$(cd "$(dirname "$PYTHON_BIN")/.." && pwd)"
export LD_LIBRARY_PATH="$PYTHON_PREFIX/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
VLLM_BIN="${VLLM_BIN:-$(dirname "$PYTHON_BIN")/vllm}"
TOOL_ARGS=()
if [[ "$ENABLE_TOOLS" == "true" || "$ENABLE_TOOLS" == "1" ]]; then
  # Qwen3.5 emits Qwen3 XML function calls.  These calls are handled only by
  # the external Analyzer bridge; the student rollout never receives tools.
  TOOL_ARGS+=(--enable-auto-tool-choice --tool-call-parser qwen3_xml)
fi

exec "$VLLM_BIN" serve "$MODEL_PATH" \
  --served-model-name "$MODEL_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --api-key "${ANALYZER_API_KEY:-local-qwen-analyzer}" \
  --dtype auto \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --enforce-eager \
  --generation-config vllm \
  "${TOOL_ARGS[@]}"
