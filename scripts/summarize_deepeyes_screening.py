#!/usr/bin/env python3
"""Summarize a judged DeepEyes V* screening run as a Markdown report."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from io import BytesIO
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image


ANSWER_PATTERN = re.compile(r"<answer>.*?</answer>", re.IGNORECASE | re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", nargs="+", type=Path, required=True)
    parser.add_argument("--judge-summary", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--examples-per-score", type=int, default=2)
    parser.add_argument("--sample-indices", nargs="*", type=int)
    return parser.parse_args()


def load_rollouts(paths: list[Path]) -> list[dict]:
    records = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            records.extend(json.loads(line) for line in handle if line.strip())
    return sorted(records, key=lambda record: int(record["index"]))


def export_images(data_path: Path, indices: set[int], image_dir: Path) -> dict[int, Path]:
    image_dir.mkdir(parents=True, exist_ok=True)
    exported: dict[int, Path] = {}
    offset = 0
    parquet = pq.ParquetFile(data_path)
    for batch in parquet.iter_batches(batch_size=128, columns=["images"]):
        rows = batch.to_pylist()
        for local_index, row in enumerate(rows):
            index = offset + local_index
            if index not in indices:
                continue
            item = row["images"][0]
            payload = item.get("bytes")
            if payload is None:
                payload = Path(item["path"]).read_bytes()
            target = (image_dir / f"question-{index:05d}.png").resolve()
            with Image.open(BytesIO(payload)) as source:
                source.convert("RGB").save(target)
            exported[index] = target
        offset += len(rows)
        if indices <= exported.keys():
            break
    return exported


def main() -> None:
    args = parse_args()
    rollouts = load_rollouts(args.rollouts)
    summary = json.loads(args.judge_summary.read_text(encoding="utf-8"))
    judged = sorted(summary["per_question"], key=lambda item: int(item["index"]))
    if len(rollouts) != len(judged):
        raise RuntimeError(f"rollout/judgement mismatch: {len(rollouts)} != {len(judged)}")

    histogram = Counter(int(item["correct_count"]) for item in judged)
    retained = [item for item in judged if int(item["correct_count"]) <= 7]
    discarded = [item for item in judged if int(item["correct_count"]) == 8]
    completions = [completion for record in rollouts for completion in record["completions"]]
    token_lengths = [len(completion["token_ids"]) for completion in completions]
    missing_tags = sum(not ANSWER_PATTERN.search(completion["text"]) for completion in completions)
    truncated = sum(completion["finish_reason"] == "length" for completion in completions)

    if args.sample_indices:
        by_index = {int(item["index"]): item for item in judged}
        missing = [index for index in args.sample_indices if index not in by_index]
        if missing:
            raise ValueError(f"sample indices not found: {missing}")
        selected = [by_index[index] for index in args.sample_indices]
    else:
        selected = []
        for score in range(9):
            selected.extend(
                [item for item in judged if int(item["correct_count"]) == score][
                    : args.examples_per_score
                ]
            )
    images = export_images(
        args.data,
        {int(item["index"]) for item in selected},
        args.output.parent / f"{args.output.stem}-images",
    )

    lines = [
        f"# Qwen3.5-2B · DeepEyes V* · {len(rollouts):,}-question screening report",
        "",
        "## Run summary",
        "",
        f"- Questions: `{len(rollouts)}`; rollouts: `{len(completions)}`; Judge: `{summary['model']}`",
        f"- Correct rollouts: `{summary['correct_rollouts']}/{summary['rollouts']}` ({summary['correct_rollouts'] / max(summary['rollouts'], 1):.2%})",
        f"- Retained by 0–7/8 rule: `{len(retained)}/{len(judged)}` ({len(retained) / max(len(judged), 1):.2%})",
        f"- Removed as 8/8: `{len(discarded)}/{len(judged)}` ({len(discarded) / max(len(judged), 1):.2%})",
        f"- Missing complete answer tags: `{missing_tags}/{len(completions)}` ({missing_tags / max(len(completions), 1):.2%})",
        f"- Length-truncated: `{truncated}/{len(completions)}` ({truncated / max(len(completions), 1):.2%})",
        f"- Generated tokens: mean `{statistics.mean(token_lengths):.1f}`, median `{statistics.median(token_lengths):.0f}`, p90 `{sorted(token_lengths)[int(0.9 * (len(token_lengths) - 1))]}`, p99 `{sorted(token_lengths)[int(0.99 * (len(token_lengths) - 1))]}`",
        f"- Judge errors: `{summary['judge_errors']}`; complete: `{summary.get('complete', False)}`",
        "",
        "## Difficulty distribution",
        "",
        "| Correct rollouts | Questions | Decision |",
        "|---:|---:|---|",
    ]
    for score in range(9):
        lines.append(f"| {score}/8 | {histogram[score]} | {'remove' if score == 8 else 'retain'} |")

    lines.extend(["", "## Representative examples", ""])
    for item in selected:
        index = int(item["index"])
        score = int(item["correct_count"])
        lines.extend(
            [
                f"### Q{index} · {score}/8",
                "",
                f"![Question {index}]({images[index]})",
                "",
                f"**Question:** {item['question']}",
                "",
                f"**Ground Truth:** {item['ground_truth']}",
                "",
                "| Rollout | Judge | Extracted answer |",
                "|---:|:---:|---|",
            ]
        )
        for rollout_index, (answer, decision) in enumerate(
            zip(item["answers"], item["judgements"], strict=True), start=1
        ):
            answer_text = " ".join(str(answer).split()).replace("|", "\\|")
            if len(answer_text) > 240:
                answer_text = answer_text[:237] + "..."
            icon = "✅" if decision == 1 else "❌" if decision == 0 else "⚠️"
            lines.append(f"| {rollout_index} | {icon} | {answer_text} |")
        lines.append("")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "retained": len(retained),
                "discarded": len(discarded),
                "histogram": {str(score): histogram[score] for score in range(9)},
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
