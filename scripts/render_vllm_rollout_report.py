#!/usr/bin/env python3
"""Render vLLM JSONL rollouts and their source images as a Markdown report."""

from __future__ import annotations

import argparse
import json
import re
from io import BytesIO
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image


ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def extract_answer(text: str) -> str | None:
    matches = ANSWER_PATTERN.findall(text)
    return matches[-1].strip() if matches else None


def load_records(paths: list[Path]) -> list[dict]:
    records: list[dict] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            records.extend(json.loads(line) for line in handle if line.strip())
    return sorted(records, key=lambda record: int(record["index"]))


def export_images(data_path: Path, indices: set[int], image_dir: Path) -> dict[int, Path]:
    image_dir.mkdir(parents=True, exist_ok=True)
    exported: dict[int, Path] = {}
    row_offset = 0
    parquet = pq.ParquetFile(data_path)
    for batch in parquet.iter_batches(batch_size=128, columns=["images"]):
        rows = batch.to_pylist()
        for local_index, row in enumerate(rows):
            index = row_offset + local_index
            if index not in indices:
                continue
            image_item = row["images"][0]
            image_bytes = image_item.get("bytes")
            if image_bytes is None:
                image_bytes = Path(image_item["path"]).read_bytes()
            output_path = (image_dir / f"question-{index:05d}.png").resolve()
            with Image.open(BytesIO(image_bytes)) as source:
                source.convert("RGB").save(output_path)
            exported[index] = output_path
        row_offset += len(rows)
        if indices <= exported.keys():
            break
    return exported


def fenced(text: str) -> str:
    return f"~~~~text\n{text.rstrip()}\n~~~~"


def main() -> None:
    args = parse_args()
    records = load_records(args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    images = export_images(
        args.data,
        {int(record["index"]) for record in records},
        args.output.parent / "images",
    )

    completion_count = sum(len(record["completions"]) for record in records)
    missing_answers = sum(
        extract_answer(completion["text"]) is None
        for record in records
        for completion in record["completions"]
    )
    truncated = sum(
        completion["finish_reason"] == "length"
        for record in records
        for completion in record["completions"]
    )

    sections = [
        "# Qwen3.5-2B · DeepEyes V* · 10-question rollout",
        "",
        "## Settings",
        "",
        "- Two independent A100 replicas; tensor parallel size: `1` per replica",
        "- Questions: `10`; rollouts per question: `8`; total completions: `80`",
        "- Sampling: temperature `1.0`, top-p `0.95`, top-k `20`",
        "- Maximum generated tokens: `512`; model context length: `8192`",
        "- Qwen chat-template thinking mode: disabled",
        "- Input: original unannotated image plus question",
        f"- Missing `<answer>` tags: `{missing_answers}/{completion_count}`; length-truncated: `{truncated}/{completion_count}`",
        "",
        "## Prompt",
        "",
        fenced(
            "<|im_start|>system\n"
            "You are a visual question-answering assistant. Reason briefly from the image and put only the final answer inside <answer>...</answer> tags.<|im_end|>\n"
            "<|im_start|>user\n"
            "<|vision_start|><|image_pad|><|vision_end|>{QUESTION}<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n\n</think>"
        ),
        "",
        "> Ground Truth is not included in the model prompt. Extracted answers below use the last complete `<answer>...</answer>` pair. No correctness judge is applied in this report.",
    ]

    for record in records:
        index = int(record["index"])
        sections.extend(
            [
                "",
                f"## Question {index}",
                "",
                f"![Question {index}]({images[index]})",
                "",
                f"**Question:** {record['question']}",
                "",
                f"**Ground Truth:** {record['ground_truth']}",
                "",
            ]
        )
        for rollout_index, completion in enumerate(record["completions"], start=1):
            answer = extract_answer(completion["text"])
            sections.extend(
                [
                    f"### Rollout {rollout_index}",
                    "",
                    f"Extracted answer: `{answer if answer is not None else '<MISSING>'}`  ",
                    f"Finish reason: `{completion['finish_reason']}` · generated tokens: `{len(completion['token_ids'])}`",
                    "",
                    fenced(completion["text"]),
                    "",
                ]
            )

    args.output.write_text("\n".join(sections).rstrip() + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "questions": len(records),
                "completions": completion_count,
                "missing_answer_tags": missing_answers,
                "length_truncated": truncated,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
