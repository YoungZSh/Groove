#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="/home/yzs/miniconda3/envs/vision-opd/bin/python"
OUTPUT_DIR="$PROJECT_ROOT/outputs/qwen35-2b-deepeyes-vstar-vllm-1000q"
mkdir -p "$OUTPUT_DIR"

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export LD_LIBRARY_PATH="/home/yzs/miniconda3/envs/vision-opd/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

run_shard() {
  local gpu="$1"
  local shard="$2"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" \
    "$PROJECT_ROOT/scripts/probe_deepeyes_vstar_vllm.py" \
    --shard-id "$shard" \
    --num-shards 2 \
    --limit 500 \
    --n 8 \
    --max-new-tokens 512 \
    --temperature 1.0 \
    --top-p 0.95 \
    --top-k 20 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 256 \
    --max-num-batched-tokens 32768 \
    --request-batch-size 64 \
    --resume \
    --output "$OUTPUT_DIR/shard-$shard.jsonl" \
    >"$OUTPUT_DIR/shard-$shard.log" 2>&1
}

# Start the remote judge first so every flushed rollout batch is scored while
# the two local GPUs continue generating the next batch.
done_marker="$OUTPUT_DIR/rollout-complete.marker"
rm -f "$done_marker"
export JUDGE_API_KEY="${JUDGE_API_KEY:-remote-qwen38}"
echo "[$(date --iso-8601=seconds)] starting streaming Qwen3.8-27B judge"
"$PYTHON" "$PROJECT_ROOT/scripts/judge_deepeyes_rollouts.py" \
  --inputs "$OUTPUT_DIR/shard-0.jsonl" "$OUTPUT_DIR/shard-1.jsonl" \
  --output "$OUTPUT_DIR/judgements-qwen38-27b.jsonl" \
  --summary "$OUTPUT_DIR/judge-summary-qwen38-27b.json" \
  --model Qwen3.8-27B \
  --base-url http://127.0.0.1:8002/v1 \
  --temperature 0.0 \
  --concurrency 128 \
  --follow \
  --done-marker "$done_marker" \
  --expected-rollouts 8000 \
  >"$OUTPUT_DIR/judge.log" 2>&1 &
judge_pid=$!

echo "[$(date --iso-8601=seconds)] starting two TP=1 rollout shards"
run_shard 0 0 &
pid0=$!
run_shard 1 1 &
pid1=$!
echo "rollout_pid0=$pid0 rollout_pid1=$pid1 judge_pid=$judge_pid"

set +e
wait "$pid0"
status0=$?
wait "$pid1"
status1=$?
set -e
echo "[$(date --iso-8601=seconds)] rollout status0=$status0 status1=$status1"
if [[ "$status0" -ne 0 || "$status1" -ne 0 ]]; then
  kill "$judge_pid" 2>/dev/null || true
  wait "$judge_pid" 2>/dev/null || true
  exit 1
fi

touch "$done_marker"
set +e
wait "$judge_pid"
judge_status=$?
set -e
if [[ "$judge_status" -ne 0 ]]; then
  echo "[$(date --iso-8601=seconds)] streaming judge exited with $judge_status; filling any missing judgements"
  "$PYTHON" "$PROJECT_ROOT/scripts/judge_deepeyes_rollouts.py" \
    --inputs "$OUTPUT_DIR/shard-0.jsonl" "$OUTPUT_DIR/shard-1.jsonl" \
    --output "$OUTPUT_DIR/judgements-qwen38-27b.jsonl" \
    --summary "$OUTPUT_DIR/judge-summary-qwen38-27b.json" \
    --model Qwen3.8-27B \
    --base-url http://127.0.0.1:8002/v1 \
    --temperature 0.0 \
    --concurrency 128 \
    >>"$OUTPUT_DIR/judge.log" 2>&1
fi
echo "[$(date --iso-8601=seconds)] rollout and judging complete"
