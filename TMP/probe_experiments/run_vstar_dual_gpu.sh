#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$ROOT/../.." && pwd)"
PYTHON=/root/siton-tmp/yzs/miniconda3/envs/vstar-glq-depo/bin/python
OUTPUT_DIR="$PROJECT_ROOT/outputs/qwen3.5-4b-vstar"
mkdir -p "$OUTPUT_DIR/logs"

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$ROOT/infer_vstar_qwen35.py" \
  --output "$OUTPUT_DIR/traces.shard-0.jsonl" \
  --num-shards 2 --shard-id 0 \
  --no-enable-thinking --no-do-sample --max-new-tokens 256 \
  >"$OUTPUT_DIR/logs/shard-0.log" 2>&1 &
PID0=$!

CUDA_VISIBLE_DEVICES=1 "$PYTHON" "$ROOT/infer_vstar_qwen35.py" \
  --output "$OUTPUT_DIR/traces.shard-1.jsonl" \
  --num-shards 2 --shard-id 1 \
  --no-enable-thinking --no-do-sample --max-new-tokens 256 \
  >"$OUTPUT_DIR/logs/shard-1.log" 2>&1 &
PID1=$!

STATUS=0
wait "$PID0" || STATUS=$?
wait "$PID1" || STATUS=$?
if [[ "$STATUS" -ne 0 ]]; then
  echo "At least one inference shard failed; inspect $OUTPUT_DIR/logs" >&2
  exit "$STATUS"
fi

# Complete any answers that reached the fast first-pass limit. Existing complete
# records are skipped, and repaired records are appended for merge-time selection.
CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$ROOT/infer_vstar_qwen35.py" \
  --output "$OUTPUT_DIR/traces.shard-0.jsonl" \
  --num-shards 2 --shard-id 0 \
  --no-enable-thinking --no-do-sample --max-new-tokens 4096 \
  --retry-truncated \
  >>"$OUTPUT_DIR/logs/shard-0.log" 2>&1 &
PID0=$!

CUDA_VISIBLE_DEVICES=1 "$PYTHON" "$ROOT/infer_vstar_qwen35.py" \
  --output "$OUTPUT_DIR/traces.shard-1.jsonl" \
  --num-shards 2 --shard-id 1 \
  --no-enable-thinking --no-do-sample --max-new-tokens 4096 \
  --retry-truncated \
  >>"$OUTPUT_DIR/logs/shard-1.log" 2>&1 &
PID1=$!

STATUS=0
wait "$PID0" || STATUS=$?
wait "$PID1" || STATUS=$?
if [[ "$STATUS" -ne 0 ]]; then
  echo "At least one retry shard failed; inspect $OUTPUT_DIR/logs" >&2
  exit "$STATUS"
fi

"$PYTHON" "$ROOT/merge_vstar_traces.py" \
  "$OUTPUT_DIR/traces.shard-0.jsonl" \
  "$OUTPUT_DIR/traces.shard-1.jsonl" \
  --output "$OUTPUT_DIR/traces.jsonl"
