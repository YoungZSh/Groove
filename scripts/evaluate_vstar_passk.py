#!/usr/bin/env python3
"""Sample current Student prompts on V*Bench and score every answer semantically.

This is separate from evaluate_vstar.py's historical greedy/rule protocol.
Generation sees only the original image and question/options. Ground truth is
retained in audit metadata and passed exclusively to the semantic Judge.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from groove.passk import compare_passk, summarize_passk


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def event(kind, **fields):
    print(json.dumps({"event": kind, **fields}, ensure_ascii=False), flush=True)


def read_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prepare(args):
    import pyarrow.parquet as pq
    from omegaconf import OmegaConf
    from transformers import AutoProcessor
    from groove.reasoning_dataset import ReasoningAnswerDataset
    from verl.utils.tokenizer import build_multimodal_processor_inputs, normalize_token_ids
    from verl.utils.tokenizer.chat_template import apply_chat_template
    from verl.workers.rollout.utils import qwen2_5_vl_dedup_image_tokens

    rows = pq.read_table(args.data).to_pylist()
    if len(rows) != 191 or len({str(r["extra_info"]["question_id"]) for r in rows}) != 191:
        raise ValueError("Expected the complete 191-question V*Bench validation split")
    if Counter(r["extra_info"]["category"] for r in rows) != {"direct_attributes": 115, "relative_position": 76}:
        raise ValueError("Unexpected V*Bench categories")
    processor = AutoProcessor.from_pretrained(str(args.processor_model or args.model), local_files_only=True)
    adapter = ReasoningAnswerDataset.__new__(ReasoningAnswerDataset)
    adapter.prompt_key = "prompt"
    adapter.image_key, adapter.video_key, adapter.audio_key = "images", "videos", "audios"
    adapter.processor, adapter.image_max_pixels = processor, None
    inputs, metadata = [], []
    for index, row in enumerate(rows):
        if index % args.num_shards != args.shard_index:
            continue
        messages = adapter._build_messages(row)
        images, videos, audios = adapter._process_multi_modal_info(
            messages, image_patch_size=16, config=OmegaConf.create({}))
        if len(images) != 1 or videos or audios:
            raise ValueError("Only the single original benchmark image is allowed")
        prompt = apply_chat_template(processor, messages, tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False)
        model_inputs = build_multimodal_processor_inputs(
            processor, text=[prompt], images=images, videos=None, audio=None, mm_processor_kwargs={})
        ids = normalize_token_ids(model_inputs.pop("input_ids"))
        if len(ids) > 9216:
            raise ValueError(f"Question {index} exceeds the training prompt limit: {len(ids)}")
        dedup = qwen2_5_vl_dedup_image_tokens(ids, processor)
        inputs.append({"prompt_token_ids": dedup, "multi_modal_data": {"image": images}})
        extra = row["extra_info"]
        image_hash = sha256_bytes(row["images"][0]["bytes"])
        if image_hash != extra["image_sha256"]:
            raise ValueError("Embedded image bytes do not match benchmark metadata")
        metadata.append({
            "question_index": index, "question_id": str(extra["question_id"]),
            "category": extra["category"], "question": extra["question"], "choices": extra["choices"],
            "ground_truth": row["reward_model"]["ground_truth"], "image_sha256": image_hash,
            "processed_image_size": list(images[0].size), "prompt": prompt,
            "prompt_sha256": sha256_bytes(json.dumps(ids).encode()), "prompt_tokens": len(ids),
            "prompt_token_ids": ids, "vllm_prompt_token_ids": dedup,
            "question_seed": args.seed + index * args.n,
        })
        if len(inputs) % 32 == 0:
            event("prepared", label=args.label, shard=args.shard_index, questions=len(inputs))
    return inputs, metadata, processor


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def generate(args):
    args.output.mkdir(parents=True, exist_ok=False)
    inputs, metadata, processor = prepare(args)
    with (args.output / "prompts.jsonl").open("x") as handle:
        for meta in metadata:
            handle.write(json.dumps(meta, ensure_ascii=False) + "\n")
    manifest = {
        "label": args.label, "model": str(args.model.resolve()),
        "processor_model": str((args.processor_model or args.model).resolve()),
        "data": str(args.data.resolve()), "data_sha256": sha256(args.data),
        "script_sha256": sha256(__file__), "chat_template_sha256": sha256_bytes(processor.chat_template.encode()),
        "n": args.n, "temperature": args.temperature, "top_p": 1.0, "top_k": -1,
        "max_tokens": args.max_tokens, "max_model_len": 10240, "seed": args.seed,
        "question_seed_formula": "seed + original_parquet_row_index * n",
        "enable_thinking": False, "enforce_eager": True, "tensor_parallel_size": 1,
        "max_num_seqs": 64, "max_num_batched_tokens": 65536, "gpu_memory_utilization": 0.45,
        "num_shards": args.num_shards, "shard_index": args.shard_index, "questions": len(inputs),
        "question_batch_size": args.question_batch_size,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "versions": {k: importlib.metadata.version(k) for k in ("torch", "vllm", "transformers", "pyarrow")},
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.prepare_only:
        event("prepare_complete", **manifest)
        return
    from vllm import LLM, SamplingParams

    event("loading", label=args.label, shard=args.shard_index, model=str(args.model))
    llm = LLM(model=str(args.model), tokenizer=str(args.processor_model or args.model),
              tensor_parallel_size=1, dtype="bfloat16", max_model_len=10240,
              max_num_seqs=64, max_num_batched_tokens=65536, enable_chunked_prefill=True,
              gpu_memory_utilization=0.45, enforce_eager=True, enable_prefix_caching=True,
              seed=args.seed, mm_processor_cache_gb=0, disable_log_stats=True, generation_config="vllm")
    start = time.monotonic()
    with (args.output / "predictions.jsonl").open("x") as handle:
        for offset in range(0, len(inputs), args.question_batch_size):
            metas = metadata[offset:offset + args.question_batch_size]
            params = [SamplingParams(temperature=args.temperature, top_p=1.0, top_k=-1,
                                     repetition_penalty=1.0, n=args.n, max_tokens=args.max_tokens,
                                     seed=meta["question_seed"]) for meta in metas]
            outputs = llm.generate(inputs[offset:offset + args.question_batch_size], params, use_tqdm=False)
            for meta, output in zip(metas, outputs, strict=True):
                if len(output.outputs) != args.n or {o.index for o in output.outputs} != set(range(args.n)):
                    raise RuntimeError("Generation returned an incomplete or duplicate sample group")
                audit = {k: v for k, v in meta.items() if k not in ("prompt_token_ids", "vllm_prompt_token_ids")}
                for completion in output.outputs:
                    record = {**audit, "model_label": args.label, "sample_index": completion.index,
                              "sample_seed": meta["question_seed"] + completion.index,
                              "output": completion.text, "response_token_ids": list(completion.token_ids),
                              "response_tokens": len(completion.token_ids), "finish_reason": completion.finish_reason}
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            event("generated", label=args.label, shard=args.shard_index,
                  questions=min(offset + args.question_batch_size, len(inputs)), total=len(inputs),
                  elapsed_seconds=round(time.monotonic() - start, 2))
    event("generation_complete", label=args.label, shard=args.shard_index,
          questions=len(inputs), samples=len(inputs) * args.n)


def judge(args):
    from groove.semantic_reward import compute_score
    import pyarrow.parquet as pq

    args.output.mkdir(parents=True, exist_ok=False)
    records = [r for path in args.inputs for r in read_jsonl(path)]
    # Interleave models by question/sample to reduce confounding by Judge service time.
    records.sort(key=lambda r: (r["question_index"], r["sample_index"], r["model_label"]))
    expected = [str(r["extra_info"]["question_id"]) for r in pq.read_table(args.data).to_pylist()]
    labels = sorted({r["model_label"] for r in records})
    if labels != ["base", "best130"]:
        raise ValueError("Expected precisely base and best130 generation records")
    coverage = {label: summarize_passk([dict(r, accuracy=0) for r in records if r["model_label"] == label],
                                       n=args.n, expected_question_ids=expected) for label in labels}
    compare_passk(coverage["base"], coverage["best130"])
    manifest = {"data_sha256": sha256(args.data), "scorer": "groove.semantic_reward.compute_score/vstar_bench",
                "scorer_sha256": sha256(Path(__file__).resolve().parents[1] / "src/groove/semantic_reward.py"),
                "judge_base_url": os.environ.get("GROOVE_JUDGE_BASE_URL", "http://127.0.0.1:8002/v1"),
                "judge_model": os.environ.get("GROOVE_JUDGE_MODEL", "Qwen3.8-27B"),
                "concurrency": args.concurrency, "n": args.n, "input_files": {str(p): sha256(p) for p in args.inputs}}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    def score(record):
        start = time.monotonic()
        result = compute_score("vstar_bench", record["output"], record["ground_truth"],
                               {"question": record["question"], "choices": record["choices"], "split": "validation"})
        if result["accuracy"] not in (0, 1) or result["score"] != result["accuracy"]:
            raise ValueError("Semantic validation must return unshaped binary accuracy")
        return {**record, **result, "judge_elapsed_seconds": time.monotonic() - start}

    scored = {label: [] for label in labels}
    handles = {label: (args.output / (label + "-scored.jsonl")).open("x") for label in labels}
    failures = []
    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = {executor.submit(score, r): r for r in records}
            for index, future in enumerate(as_completed(futures), 1):
                try:
                    result = future.result()
                    label = result["model_label"]
                    scored[label].append(result)
                    handles[label].write(json.dumps(result, ensure_ascii=False) + "\n")
                    handles[label].flush()
                except Exception as error:
                    record = futures[future]
                    failures.append({"model_label": record["model_label"], "question_id": record["question_id"],
                                     "sample_index": record["sample_index"], "error_type": type(error).__name__})
                if index % 64 == 0 or index == len(records):
                    event("scored", completed=index, total=len(records), failures=len(failures))
    finally:
        for handle in handles.values():
            handle.close()
    if failures:
        (args.output / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
        raise RuntimeError("Judge failures remain; pass@k is not computed on incomplete results")
    summaries = {label: summarize_passk(rows, n=args.n, expected_question_ids=expected)
                 for label, rows in scored.items()}
    summaries["comparison"] = compare_passk(summaries["base"], summaries["best130"])
    for label in labels:
        rows = scored[label]
        summaries[label]["diagnostics"] = {
            "format_valid_fraction": sum(r["format_valid"] for r in rows) / len(rows),
            "rule_accuracy": sum(r["rule_accuracy"] for r in rows) / len(rows),
            "rule_unparsed_fraction": sum(r["rule_unparsed"] for r in rows) / len(rows),
            "length_truncated_fraction": sum(r["finish_reason"] == "length" for r in rows) / len(rows),
            "mean_response_tokens": sum(r["response_tokens"] for r in rows) / len(rows),
        }
    (args.output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    event("judge_complete", comparison=summaries["comparison"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--model", type=Path, required=True)
    gen.add_argument("--processor-model", type=Path)
    gen.add_argument("--label", choices=["base", "best130"], required=True)
    gen.add_argument("--shard-index", type=int, default=0)
    gen.add_argument("--num-shards", type=int, default=2)
    gen.add_argument("--seed", type=int, default=20260904)
    gen.add_argument("--temperature", type=float, default=1.0)
    gen.add_argument("--max-tokens", type=int, default=1024)
    gen.add_argument("--question-batch-size", type=int, default=16)
    gen.add_argument("--prepare-only", action="store_true")
    score = sub.add_parser("judge")
    score.add_argument("--inputs", nargs="+", type=Path, required=True)
    score.add_argument("--concurrency", type=int, default=32)
    for command in (gen, score):
        command.add_argument("--data", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--n", type=int, default=8)
    args = parser.parse_args()
    if args.n < 1:
        parser.error("n must be positive")
    if args.command == "generate":
        if not 0 <= args.shard_index < args.num_shards or args.question_batch_size < 1 or args.temperature <= 0:
            parser.error("Sampling requires a valid shard, positive batch size and positive temperature")
        generate(args)
    else:
        if args.concurrency < 1:
            parser.error("concurrency must be positive")
        judge(args)


if __name__ == "__main__":
    main()
