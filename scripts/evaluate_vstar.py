#!/usr/bin/env python3
"""Evaluate a merged Student checkpoint on the original V*Bench test parquet.

Uses original image bytes, the historical DeepEyes Student system instruction,
and the checkpoint's chat template. No Teacher, external tools, or LLM judge.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import re
import time


SYSTEM_PROMPT = (
    "You are a visual question-answering assistant. Analyze the image and answer "
    "the question. Put only the final answer inside <answer>...</answer> tags."
)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
TAG_RE = re.compile(r"</?\s*answer\b[^>]*>", re.IGNORECASE)
CHOICE_RE = re.compile(r"^\(([A-D])\)\s*(.+)$", re.MULTILINE)


def question_text(text: str) -> str:
    # Replace only the source's output-format instruction; retain question/options.
    question = re.sub(
        r"\nAnswer with the option(?:'s)? letter.*$", "", text.strip(), flags=re.I
    )
    return question + "\nReturn the selected option letter inside <answer>...</answer>."


def normalized(text: str) -> str:
    return " ".join(text.strip().strip("*` ").split()).rstrip(".! ").casefold()


def parse_prediction(output: str, choices: dict[str, str]) -> dict:
    """Parse a final answer without looking for arbitrary letters in reasoning.

    Accept an isolated option label, a label followed by its matching option
    text, or an exact unambiguous option text. Ambiguous finals remain unparsed.
    """
    matches = list(ANSWER_RE.finditer(output))
    candidate = matches[-1].group(1).strip() if matches else output.strip()
    tags = TAG_RE.findall(output)
    format_valid = bool(
        len(matches) == 1 and candidate and tags == ["<answer>", "</answer>"]
        and not output[matches[-1].end():].strip()
    )
    # Without a complete tag, only use an explicitly marked final line or the
    # full short response; never guess from letters discussed in the rationale.
    if not matches and "\n" in candidate:
        last = candidate.splitlines()[-1].strip()
        if re.match(r"(?i)^(?:final(?: answer)?|answer)\s*[:：]", last):
            candidate = last
    cleaned = candidate.strip().strip("*` ")
    cleaned = re.sub(r"(?i)^(?:(?:the\s+)?(?:final\s+)?answer|option)\s*(?:is\s*)?[:：]?\s*", "", cleaned)
    label = re.fullmatch(r"\(?([A-D])\)?(?:[.:：\-]\s*|\s+)?(.*?)", cleaned, re.DOTALL)
    prediction = None
    method = "unparsed"
    if label and label.group(1) in choices:
        suffix = label.group(2).strip()
        if not suffix or normalized(suffix) == normalized(choices[label.group(1)]):
            prediction = label.group(1)
            method = "option_label"
    if prediction is None:
        equivalent = [key for key, value in choices.items() if normalized(candidate) == normalized(value)]
        if len(equivalent) == 1:
            prediction = equivalent[0]
            method = "exact_option_text"
    return {
        "answer_text": candidate, "predicted_label": prediction,
        "parse_method": method, "format_valid": format_valid,
        "has_complete_answer_tag": bool(matches),
    }


def summary(rows: list[dict]) -> dict:
    def rates(items):
        n = len(items)
        correct = sum(row["correct"] for row in items)
        return {
            "count": n, "correct": correct, "accuracy": correct / n,
            "format_valid": sum(row["format_valid"] for row in items),
            "correct_with_valid_format": sum(row["correct"] and row["format_valid"] for row in items),
            "unparsed": sum(row["predicted_label"] is None for row in items),
            "length_truncated": sum(row["finish_reason"] == "length" for row in items),
            "mean_response_tokens": sum(row["response_tokens"] for row in items) / n,
            "parse_methods": dict(Counter(row["parse_method"] for row in items)),
        }
    return {
        "overall": rates(rows),
        "categories": {key: rates([row for row in rows if row["category"] == key])
                       for key in sorted({row["category"] for row in rows})},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint-source", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.65)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("batch-size must be positive")
    # A new output directory is required to avoid silently mixing evaluations.
    args.output.mkdir(parents=True, exist_ok=False)

    import importlib.metadata
    import pyarrow.parquet as pq
    from PIL import Image
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    source = pq.read_table(args.data).to_pylist()
    assert len(source) == 191, f"Expected full test split, found {len(source)}"
    assert len({row["question_id"] for row in source}) == len(source)
    assert Counter(row["category"] for row in source) == {"direct_attributes": 115, "relative_position": 76}
    assert all(row["image"].get("bytes") for row in source)
    processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True)
    template = processor.chat_template
    assert isinstance(template, str)
    manifest = {
        "model": str(args.model.resolve()), "checkpoint_source": str(args.checkpoint_source.resolve()),
        "training_run_id": args.run_id, "checkpoint_step": 123,
        "dataset": str(args.data.resolve()), "dataset_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "rows": len(source), "system_prompt": SYSTEM_PROMPT,
        "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
        "image_processor": processor.image_processor.to_dict(),
        "image_policy": "original embedded image bytes; native checkpoint processor resizing",
        "temperature": 0.0, "top_p": 1.0, "top_k": -1, "n": 1,
        "max_new_tokens": args.max_new_tokens, "max_model_len": args.max_model_len,
        "seed": args.seed, "enable_thinking": False, "batch_size": args.batch_size,
        "scorer": "rule-based option label or exact option text; unparsed counts wrong",
        "tools_enabled": False, "teacher_enabled": False,
        "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "vllm", "transformers", "pyarrow")},
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print(json.dumps({"event": "loading", "model": str(args.model), "run_id": args.run_id}), flush=True)
    start = time.monotonic()
    llm = LLM(
        model=str(args.model), tensor_parallel_size=1, dtype="bfloat16",
        max_model_len=args.max_model_len, max_num_seqs=args.batch_size,
        max_num_batched_tokens=8192, enable_chunked_prefill=True,
        gpu_memory_utilization=args.gpu_memory_utilization, enforce_eager=True,
        skip_mm_profiling=True, limit_mm_per_prompt={"image": 1},
        seed=args.seed, disable_log_stats=True,
    )
    sampling = SamplingParams(temperature=0.0, top_p=1.0, top_k=-1, n=1,
                              max_tokens=args.max_new_tokens, seed=args.seed)
    records = []
    with (args.output / "predictions.jsonl").open("x") as handle:
        for offset in range(0, len(source), args.batch_size):
            batch = source[offset:offset + args.batch_size]
            inputs, metadata = [], []
            for row in batch:
                image_bytes = row["image"]["bytes"]
                image = Image.open(BytesIO(image_bytes)).convert("RGB")
                choices = dict(CHOICE_RE.findall(row["text"]))
                assert str(row["label"]).upper() in choices
                question = question_text(row["text"])
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]},
                ]
                prompt = processor.apply_chat_template(messages, tokenize=False,
                                                       add_generation_prompt=True, enable_thinking=False)
                inputs.append({"prompt": prompt, "multi_modal_data": {"image": image}})
                metadata.append({
                    "question_id": str(row["question_id"]), "category": row["category"],
                    "question": question, "choices": choices, "ground_truth": str(row["label"]).upper(),
                    "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
                    "image_size": list(image.size), "prompt": prompt,
                })
            outputs = llm.generate(inputs, sampling, use_tqdm=False)
            for meta, output in zip(metadata, outputs, strict=True):
                completion = output.outputs[0]
                parsed = parse_prediction(completion.text, meta["choices"])
                record = {
                    **meta, **parsed, "output": completion.text,
                    "correct": parsed["predicted_label"] == meta["ground_truth"],
                    "response_token_ids": list(completion.token_ids),
                    "response_tokens": len(completion.token_ids),
                    "prompt_tokens": len(output.prompt_token_ids),
                    "finish_reason": completion.finish_reason,
                }
                records.append(record)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            print(json.dumps({"event": "progress", "completed": len(records), "total": len(source),
                              "elapsed_s": round(time.monotonic() - start, 1)}), flush=True)
    result = summary(records)
    result["elapsed_s"] = time.monotonic() - start
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"event": "complete", **result}), flush=True)


if __name__ == "__main__":
    main()
