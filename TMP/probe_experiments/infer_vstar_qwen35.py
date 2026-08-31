#!/usr/bin/env python3
"""Run Qwen3.5-4B on V*Bench and save full per-example inference traces."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
import transformers
from PIL import Image
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration


DEFAULT_MODEL = Path("/root/siton-tmp/yzs/ckpts/Qwen3.5-4B")
DEFAULT_DATA = Path(
    "/root/siton-tmp/yzs/datasets/vstar-bench/data/test-00000-of-00001.parquet"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--attn-implementation",
        default="flash_attention_2",
        choices=("flash_attention_2", "sdpa", "eager"),
    )
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Qwen's explicit <think> reasoning mode (disabled by default).",
    )
    parser.add_argument(
        "--do-sample",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable sampling; deterministic greedy decoding is the default.",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--retry-truncated",
        action="store_true",
        help="When resuming, rerun records that hit the previous token limit.",
    )
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.model.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {args.model}")
    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset file does not exist: {args.data}")
    if args.num_shards < 1:
        raise ValueError("--num-shards must be at least 1")
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("--shard-id must be in [0, --num-shards)")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")


def load_completed_indices(output: Path, retry_truncated: bool) -> set[int]:
    completed: set[int] = set()
    if not output.exists():
        return completed
    with output.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL in {output} at line {line_number}; "
                    "move or repair the file before resuming"
                ) from exc
            if "error" not in record and not (retry_truncated and record.get("truncated")):
                completed.add(int(record["index"]))
    return completed


def split_trace(completion: str, enable_thinking: bool) -> tuple[str, str]:
    text = completion.strip()
    if not enable_thinking:
        return "", text
    if text.startswith("<think>"):
        text = text[len("<think>") :].lstrip()
    if "</think>" not in text:
        return text, ""
    reasoning, final_answer = text.split("</think>", maxsplit=1)
    return reasoning.strip(), final_answer.strip()


def extract_label(final_answer: str, completion: str) -> str | None:
    candidates = [final_answer, completion]
    patterns = (
        r"(?:correct\s+answer|answer|答案)\s*(?:is|为|是|[:：])?"
        r"\s*[*_`]*\(?([A-D])\)?",
        r"^\s*[*_`]*\(?([A-D])\)?[*_`]*(?:\s|[.、:：]|$)",
        r"\(([A-D])\)[^\n]{0,100}(?:✅\s*)?correct\b",
        r"[*_`]*\(?([A-D])\)?[*_`]*[.。]?\s*$",
    )
    for text in candidates:
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).upper()
    return None


def eos_token_ids(model: Qwen3_5ForConditionalGeneration) -> set[int]:
    value = model.generation_config.eos_token_id
    if value is None:
        return set()
    if isinstance(value, int):
        return {value}
    return {int(token_id) for token_id in value}


def append_record(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def main() -> int:
    args = parse_args()
    validate_args(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    completed = load_completed_indices(args.output, args.retry_truncated)
    table = pq.read_table(args.data)
    rows = table.to_pylist()
    selected = [
        index
        for index in range(len(rows))
        if index % args.num_shards == args.shard_id and index not in completed
    ]
    if args.limit is not None:
        selected = selected[: args.limit]

    print(
        f"dataset_rows={len(rows)} shard={args.shard_id}/{args.num_shards} "
        f"already_completed={len(completed)} pending={len(selected)}",
        flush=True,
    )
    if not selected:
        return 0

    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        local_files_only=True,
    ).eval()
    model.to(args.device)
    eos_ids = eos_token_ids(model)

    run_metadata = {
        "model": str(args.model),
        "dataset": str(args.data),
        "device": args.device,
        "gpu": torch.cuda.get_device_name(torch.device(args.device)),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "python_version": platform.python_version(),
        "dtype": "bfloat16",
        "attention_implementation": args.attn_implementation,
        "enable_thinking": args.enable_thinking,
        "do_sample": args.do_sample,
        "temperature": args.temperature if args.do_sample else None,
        "top_p": args.top_p if args.do_sample else None,
        "top_k": args.top_k if args.do_sample else None,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
    }

    started = time.perf_counter()
    with args.output.open("a", encoding="utf-8") as output_handle:
        for position, index in enumerate(selected, start=1):
            row = rows[index]
            item_started = time.perf_counter()
            try:
                image_data = row["image"]
                image = Image.open(BytesIO(image_data["bytes"])).convert("RGB")
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": row["text"]},
                        ],
                    }
                ]
                prompt = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=args.enable_thinking,
                )
                inputs = processor(
                    text=[prompt],
                    images=[image],
                    return_tensors="pt",
                ).to(args.device)
                input_tokens = int(inputs["input_ids"].shape[-1])

                generation_args: dict[str, Any] = {
                    "max_new_tokens": args.max_new_tokens,
                    "do_sample": args.do_sample,
                    "use_cache": True,
                }
                if args.do_sample:
                    torch.manual_seed(args.seed + index)
                    torch.cuda.manual_seed_all(args.seed + index)
                    generation_args.update(
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                    )

                torch.cuda.synchronize(torch.device(args.device))
                generation_started = time.perf_counter()
                with torch.inference_mode():
                    generated = model.generate(**inputs, **generation_args)
                torch.cuda.synchronize(torch.device(args.device))
                generation_seconds = time.perf_counter() - generation_started

                token_ids = generated[0, input_tokens:]
                token_id_list = token_ids.detach().cpu().tolist()
                raw_completion = processor.decode(token_ids, skip_special_tokens=False)
                completion = processor.decode(token_ids, skip_special_tokens=True)
                reasoning_trace, final_answer = split_trace(
                    completion, args.enable_thinking
                )
                predicted_label = extract_label(final_answer, completion)
                stopped_on_eos = bool(token_id_list and token_id_list[-1] in eos_ids)

                record = {
                    "index": index,
                    "question_id": row["question_id"],
                    "category": row["category"],
                    "image_path": image_data.get("path"),
                    "image_width": image.width,
                    "image_height": image.height,
                    "question": row["text"],
                    "ground_truth_label": row["label"],
                    "predicted_label": predicted_label,
                    "correct": predicted_label == row["label"] if predicted_label else None,
                    "prompt": prompt,
                    "reasoning_trace": reasoning_trace,
                    "final_answer": final_answer,
                    "completion": completion,
                    "raw_completion": raw_completion,
                    "generated_token_ids": token_id_list,
                    "input_tokens": input_tokens,
                    "output_tokens": len(token_id_list),
                    "generation_seconds": generation_seconds,
                    "tokens_per_second": (
                        len(token_id_list) / generation_seconds if generation_seconds else None
                    ),
                    "stopped_on_eos": stopped_on_eos,
                    "truncated": len(token_id_list) >= args.max_new_tokens and not stopped_on_eos,
                    "run": run_metadata,
                }
                append_record(output_handle, record)
                status = "correct" if record["correct"] else "wrong"
                print(
                    f"[{position}/{len(selected)}] index={index} qid={row['question_id']} "
                    f"pred={predicted_label} gold={row['label']} {status} "
                    f"in={input_tokens} out={len(token_id_list)} "
                    f"time={generation_seconds:.1f}s",
                    flush=True,
                )
            except Exception as exc:
                error_record = {
                    "index": index,
                    "question_id": row.get("question_id"),
                    "category": row.get("category"),
                    "error": f"{type(exc).__name__}: {exc}",
                    "elapsed_seconds": time.perf_counter() - item_started,
                    "run": run_metadata,
                }
                append_record(output_handle, error_record)
                print(
                    f"[{position}/{len(selected)}] index={index} ERROR {error_record['error']}",
                    file=sys.stderr,
                    flush=True,
                )
                if args.fail_fast:
                    raise

    print(
        f"shard={args.shard_id} finished elapsed={time.perf_counter() - started:.1f}s "
        f"output={args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
