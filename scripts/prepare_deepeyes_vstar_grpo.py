#!/usr/bin/env python3
"""Build reproducible train/validation splits from screened DeepEyes V* rows."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data/deepeyes_47k/data_0.1.2_visual_toolbox_v2.parquet"
DEFAULT_JUDGE_SUMMARY = (
    ROOT
    / "outputs/qwen35-2b-deepeyes-vstar-vllm-full/judge-summary-qwen38-27b.json"
)
SYSTEM_PROMPT = (
    "You are a visual question-answering assistant. "
    "Analyze the image and answer the question. "
    "Put only the final answer inside <answer>...</answer> tags."
)
# Manually audited source rows whose question and reference answer contradict
# one another. Keep this exclusion in the reproducible builder so regenerating
# the split cannot silently reintroduce the bad reward target.
DEFAULT_EXCLUDED_SOURCE_INDICES = (10681,)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--judge-summary", type=Path, default=DEFAULT_JUDGE_SUMMARY)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=2200)
    parser.add_argument("--validation-size", type=int, default=220)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--max-correct", type=int, default=7)
    parser.add_argument(
        "--exclude-source-index",
        action="append",
        type=int,
        default=list(DEFAULT_EXCLUDED_SOURCE_INDICES),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_selected_rows(source: Path, selected: set[int]) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    offset = 0
    parquet = pq.ParquetFile(source)
    for batch in parquet.iter_batches(batch_size=128, columns=["images", "extra_info"]):
        batch_rows = batch.to_pylist()
        for local_index, row in enumerate(batch_rows):
            source_index = offset + local_index
            if source_index in selected:
                rows[source_index] = row
        offset += len(batch_rows)
    missing = selected - rows.keys()
    if missing:
        raise RuntimeError(f"failed to load {len(missing)} selected source rows")
    return rows


def image_payload(row: dict) -> bytes:
    image = row["images"][0]
    payload = image.get("bytes")
    if payload is None:
        payload = Path(image["path"]).read_bytes()
    return payload


def image_group_split(
    selected_indices: list[int],
    rows: dict[int, dict],
    validation_size: int,
    seed: int,
) -> tuple[list[int], list[int], dict[int, str]]:
    hashes = {
        index: hashlib.sha256(image_payload(rows[index])).hexdigest()
        for index in selected_indices
    }
    grouped: dict[str, list[int]] = collections.defaultdict(list)
    for index in selected_indices:
        grouped[hashes[index]].append(index)

    groups = list(grouped.values())
    rng = random.Random(seed + 1)
    rng.shuffle(groups)

    # Randomized subset-sum over image groups gives an exact validation size
    # without allowing the same source image to cross the split boundary.
    choices: dict[int, list[int]] = {0: []}
    for group_index, group in enumerate(groups):
        group_size = len(group)
        for count in sorted(list(choices), reverse=True):
            new_count = count + group_size
            if new_count <= validation_size and new_count not in choices:
                choices[new_count] = choices[count] + [group_index]
        if validation_size in choices:
            break
    if validation_size not in choices:
        raise RuntimeError(f"could not construct an image-grouped validation split of {validation_size}")

    validation_groups = set(choices[validation_size])
    validation = [index for i, group in enumerate(groups) if i in validation_groups for index in group]
    validation_set = set(validation)
    train = [index for index in selected_indices if index not in validation_set]
    rng.shuffle(train)
    rng.shuffle(validation)

    train_hashes = {hashes[index] for index in train}
    validation_hashes = {hashes[index] for index in validation}
    if train_hashes & validation_hashes:
        raise RuntimeError("image leakage detected across train and validation")
    return train, validation, hashes


def build_record(
    source_index: int,
    row: dict,
    split: str,
    correct_count: int,
    image_sha256: str,
) -> dict:
    question = str(row["extra_info"]["question"]).strip()
    ground_truth = str(row["extra_info"]["answer"]).strip()
    return {
        "data_source": "deepeyes_vstar_grpo",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"<image>{question}"},
        ],
        # These are the original embedded source images. No boxes, labels, or
        # privileged crops are added to the Student input.
        "images": row["images"],
        "ability": "visual_question_answering",
        "reward_model": {"style": "model", "ground_truth": ground_truth},
        "extra_info": {
            "answer": ground_truth,
            "question": question,
            "index": str(source_index),
            "question_id": f"deepeyes-vstar-{source_index:06d}",
            "source_index": source_index,
            "split": split,
            "screening_correct_count": correct_count,
            "screening_rollouts": 8,
            "screening_judge": "Qwen3.8-27B",
            "image_sha256": image_sha256,
        },
    }


def score_counts(indices: list[int], scores: dict[int, int]) -> dict[str, int]:
    counts = collections.Counter(scores[index] for index in indices)
    return {str(score): counts[score] for score in range(9) if counts[score]}


def main() -> None:
    args = parse_args()
    if args.sample_size <= 0 or args.validation_size <= 0:
        raise ValueError("sample-size and validation-size must be positive")
    if args.validation_size >= args.sample_size:
        raise ValueError("validation-size must be smaller than sample-size")
    if not 0 <= args.max_correct <= 8:
        raise ValueError("max-correct must be in [0, 8]")

    source = args.source.resolve()
    judge_summary_path = args.judge_summary.resolve()
    output_dir = args.output_dir.resolve()
    summary = json.loads(judge_summary_path.read_text(encoding="utf-8"))
    if not summary.get("complete") or int(summary.get("judge_errors", -1)) != 0:
        raise RuntimeError("judge summary must be complete and error-free")

    scores = {
        int(item["index"]): int(item["correct_count"])
        for item in summary["per_question"]
        if int(item["judged_count"]) == 8
    }
    candidates = sorted(index for index, score in scores.items() if score <= args.max_correct)
    if len(candidates) < args.sample_size:
        raise ValueError(f"only {len(candidates)} candidates for sample-size={args.sample_size}")

    rng = random.Random(args.seed)
    selected = rng.sample(candidates, args.sample_size)
    rows = load_selected_rows(source, set(selected))
    train_indices, validation_indices, image_hashes = image_group_split(
        selected, rows, args.validation_size, args.seed
    )
    selected_exclusions = set(selected) & set(args.exclude_source_index)
    train_indices = [index for index in train_indices if index not in selected_exclusions]
    validation_indices = [index for index in validation_indices if index not in selected_exclusions]

    train_records = [
        build_record(index, rows[index], "train", scores[index], image_hashes[index])
        for index in train_indices
    ]
    validation_records = [
        build_record(index, rows[index], "validation", scores[index], image_hashes[index])
        for index in validation_indices
    ]
    final_sample_size = args.sample_size - len(selected_exclusions)
    if len(train_records) + len(validation_records) != final_sample_size:
        raise RuntimeError("split size mismatch")

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.parquet"
    validation_path = output_dir / "validation.parquet"
    pq.write_table(pa.Table.from_pylist(train_records), train_path, compression="zstd")
    pq.write_table(pa.Table.from_pylist(validation_records), validation_path, compression="zstd")

    manifest = {
        "source": str(source),
        "source_sha256": sha256_file(source),
        "judge_summary": str(judge_summary_path),
        "judge_summary_sha256": sha256_file(judge_summary_path),
        "seed": args.seed,
        "candidate_rule": f"0 <= correct_count <= {args.max_correct} out of 8",
        "candidate_rows": len(candidates),
        "requested_sample_size": args.sample_size,
        "sample_size": final_sample_size,
        "excluded_source_indices": sorted(selected_exclusions),
        "prompt": SYSTEM_PROMPT,
        "thinking_enabled": False,
        "student_images": "original embedded images; no annotations or privileged crops",
        "image_grouped_split": True,
        "train": {
            "path": str(train_path),
            "rows": len(train_records),
            "score_distribution": score_counts(train_indices, scores),
        },
        "validation": {
            "path": str(validation_path),
            "rows": len(validation_records),
            "score_distribution": score_counts(validation_indices, scores),
        },
        "selected_source_indices": {
            "train": train_indices,
            "validation": validation_indices,
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "selected_source_indices"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
