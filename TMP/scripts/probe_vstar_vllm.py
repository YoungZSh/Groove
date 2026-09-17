#!/usr/bin/env python3
"""Run a bounded Qwen3.5 vLLM sampling probe on the V* data."""

from __future__ import annotations

import argparse
import json
import os
import time
from io import BytesIO
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = Path("/root/siton-tmp/yzs/ckpts/Qwen3.5-2B")
DEFAULT_DATA = ROOT / "data/visual_toolbox_47k/data_0.1.2_visual_toolbox_v2.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--request-batch-size", type=int, default=64)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=20260903)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.model.is_dir():
        raise FileNotFoundError(args.model)
    if not args.data.is_file():
        raise FileNotFoundError(args.data)
    if args.num_shards <= 0 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError("shard-id must be in [0, num-shards)")
    positive = {
        "limit": args.limit,
        "n": args.n,
        "max-new-tokens": args.max_new_tokens,
        "max-num-seqs": args.max_num_seqs,
        "max-num-batched-tokens": args.max_num_batched_tokens,
        "request-batch-size": args.request_batch_size,
    }
    if any(value <= 0 for value in positive.values()):
        raise ValueError(f"these arguments must be positive: {positive}")


def load_rows(args: argparse.Namespace) -> list[tuple[int, dict]]:
    selected: list[tuple[int, dict]] = []
    offset = 0
    parquet = pq.ParquetFile(args.data)
    for batch in parquet.iter_batches(batch_size=128, columns=["images", "extra_info"]):
        rows = batch.to_pylist()
        for local_index, row in enumerate(rows):
            index = offset + local_index
            if index % args.num_shards != args.shard_id:
                continue
            selected.append((index, row))
            if len(selected) == args.limit:
                return selected
        offset += len(rows)
    return selected


def build_inputs(processor, selected: list[tuple[int, dict]]) -> tuple[list[dict], list[dict]]:
    prompts: list[dict] = []
    metadata: list[dict] = []
    for index, row in selected:
        image_item = row["images"][0]
        image_bytes = image_item.get("bytes")
        if image_bytes is None:
            image_bytes = Path(image_item["path"]).read_bytes()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        extra = row["extra_info"]
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a visual question-answering assistant. "
                    "Reason briefly from the image and put only the final answer "
                    "inside <answer>...</answer> tags."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": str(extra["question"])},
                ],
            },
        ]
        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompts.append({"prompt": prompt, "multi_modal_data": {"image": image}})
        metadata.append(
            {
                "index": index,
                "question": str(extra["question"]),
                "ground_truth": str(extra["answer"]),
                "image_width": image.width,
                "image_height": image.height,
            }
        )
    return prompts, metadata


def completed_indices(path: Path) -> set[int]:
    if not path.exists():
        return set()
    completed: set[int] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                completed.add(int(json.loads(line)["index"]))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(f"invalid resume record at {path}:{line_number}") from exc
    return completed


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    selected = load_rows(args)
    if len(selected) != args.limit:
        raise RuntimeError(f"requested {args.limit} rows but found {len(selected)}")
    completed = completed_indices(args.output) if args.resume else set()
    selected = [(index, row) for index, row in selected if index not in completed]
    if not selected:
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "requested_prompts": args.limit,
                    "resumed_prompts": len(completed),
                    "remaining_prompts": 0,
                }
            ),
            flush=True,
        )
        return
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)

    load_started = time.perf_counter()
    model = LLM(
        model=str(args.model),
        tensor_parallel_size=1,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enable_chunked_prefill=True,
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        limit_mm_per_prompt={"image": 1},
        # Qwen3.5's default dummy image profiles the maximum vision-token
        # budget (16K tokens), which makes engine startup CPU-bound for several
        # minutes.  The two A100s have ample headroom for this 2B smoke test,
        # so skip that conservative activation probe.
        skip_mm_profiling=True,
        disable_log_stats=True,
        seed=args.seed + args.shard_id,
    )
    load_seconds = time.perf_counter() - load_started
    sampling = SamplingParams(
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_new_tokens,
        seed=args.seed + args.shard_id,
    )

    generation_started = time.perf_counter()
    generated_tokens = 0
    generated_prompts = 0
    output_mode = "a" if args.resume and args.output.exists() else "w"
    with args.output.open(output_mode, encoding="utf-8") as handle:
        for batch_start in range(0, len(selected), args.request_batch_size):
            batch_rows = selected[batch_start : batch_start + args.request_batch_size]
            prompts, metadata = build_inputs(processor, batch_rows)
            outputs = model.generate(prompts, sampling, use_tqdm=True)
            batch_tokens = sum(
                len(item.token_ids) for output in outputs for item in output.outputs
            )
            generated_tokens += batch_tokens
            generated_prompts += len(outputs)
            for meta, output in zip(metadata, outputs, strict=True):
                record = {
                    **meta,
                    "shard_id": args.shard_id,
                    "tensor_parallel_size": 1,
                    "n": args.n,
                    "completions": [
                        {
                            "text": item.text,
                            "token_ids": list(item.token_ids),
                            "finish_reason": item.finish_reason,
                        }
                        for item in output.outputs
                    ],
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            elapsed = time.perf_counter() - generation_started
            print(
                json.dumps(
                    {
                        "event": "batch_complete",
                        "shard_id": args.shard_id,
                        "generated_prompts": generated_prompts,
                        "remaining_prompts": len(selected) - generated_prompts,
                        "generated_tokens": generated_tokens,
                        "generation_seconds": elapsed,
                        "tokens_per_second": generated_tokens / max(elapsed, 1e-9),
                    }
                ),
                flush=True,
            )
    generation_seconds = time.perf_counter() - generation_started

    print(
        json.dumps(
            {
                "output": str(args.output),
                "model": str(args.model),
                "tensor_parallel_size": 1,
                "prompts": generated_prompts,
                "resumed_prompts": len(completed),
                "samples_per_prompt": args.n,
                "max_num_seqs": args.max_num_seqs,
                "max_num_batched_tokens": args.max_num_batched_tokens,
                "request_batch_size": args.request_batch_size,
                "generated_tokens": generated_tokens,
                "load_seconds": load_seconds,
                "generation_seconds": generation_seconds,
                "tokens_per_second": generated_tokens / max(generation_seconds, 1e-9),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
