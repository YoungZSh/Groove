#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="/home/yzs/miniconda3/envs/vision-opd/bin/python"
SEED_DIR="$PROJECT_ROOT/outputs/qwen35-2b-vstar-vllm-1000q"
OUTPUT_DIR="$PROJECT_ROOT/outputs/qwen35-2b-vstar-vllm-full"
TOTAL_QUESTIONS=22362
QUESTIONS_PER_SHARD=11181
TOTAL_ROLLOUTS=178896
mkdir -p "$OUTPUT_DIR"

seed_file() {
  local name="$1"
  if [[ ! -f "$OUTPUT_DIR/$name" ]]; then
    cp --reflink=auto "$SEED_DIR/$name" "$OUTPUT_DIR/$name"
  fi
}

# Seed the full run with the validated 0..999 results. Subsequent restarts append
# only unseen question indices and unseen (question, rollout) judge pairs.
seed_file shard-0.jsonl
seed_file shard-1.jsonl
seed_file judgements-qwen38-27b.jsonl
seed_file judge-summary-qwen38-27b.json

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export LD_LIBRARY_PATH="/home/yzs/miniconda3/envs/vision-opd/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export JUDGE_API_KEY="${JUDGE_API_KEY:-remote-qwen38}"

run_shard() {
  local gpu="$1"
  local shard="$2"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" \
    "$PROJECT_ROOT/TMP/scripts/probe_vstar_vllm.py" \
    --shard-id "$shard" \
    --num-shards 2 \
    --limit "$QUESTIONS_PER_SHARD" \
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

done_marker="$OUTPUT_DIR/rollout-complete.marker"
rm -f "$done_marker"
echo "[$(date --iso-8601=seconds)] starting streaming Qwen3.8-27B judge"
"$PYTHON" "$PROJECT_ROOT/TMP/scripts/judge_vstar_rollouts.py" \
  --inputs "$OUTPUT_DIR/shard-0.jsonl" "$OUTPUT_DIR/shard-1.jsonl" \
  --output "$OUTPUT_DIR/judgements-qwen38-27b.jsonl" \
  --summary "$OUTPUT_DIR/judge-summary-qwen38-27b.json" \
  --model Qwen3.8-27B \
  --base-url http://127.0.0.1:8002/v1 \
  --temperature 0.0 \
  --concurrency 128 \
  --batch-size 512 \
  --summary-every-rollouts 4096 \
  --follow \
  --done-marker "$done_marker" \
  --expected-rollouts "$TOTAL_ROLLOUTS" \
  >"$OUTPUT_DIR/judge.log" 2>&1 &
judge_pid=$!

echo "[$(date --iso-8601=seconds)] continuing V* rollouts after the first 1000 questions"
run_shard 0 0 &
pid0=$!
run_shard 1 1 &
pid1=$!
echo "rollout_pid0=$pid0 rollout_pid1=$pid1 judge_pid=$judge_pid total_questions=$TOTAL_QUESTIONS"

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
  echo "[$(date --iso-8601=seconds)] streaming judge exited with $judge_status; filling missing judgements"
  "$PYTHON" "$PROJECT_ROOT/TMP/scripts/judge_vstar_rollouts.py" \
    --inputs "$OUTPUT_DIR/shard-0.jsonl" "$OUTPUT_DIR/shard-1.jsonl" \
    --output "$OUTPUT_DIR/judgements-qwen38-27b.jsonl" \
    --summary "$OUTPUT_DIR/judge-summary-qwen38-27b.json" \
    --model Qwen3.8-27B \
    --base-url http://127.0.0.1:8002/v1 \
    --temperature 0.0 \
    --concurrency 128 \
    --batch-size 512 \
    --summary-every-rollouts 4096 \
    >>"$OUTPUT_DIR/judge.log" 2>&1
fi
echo "[$(date --iso-8601=seconds)] full V* rollout and judging complete"
