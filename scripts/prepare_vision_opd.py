#!/usr/bin/env python3
"""Build leakage-free 95/5 verl splits from the local Vision-OPD-6K data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

from datasets import Dataset


DEFAULT_SOURCE = Path("/root/siton-tmp/yzs/Vision-OPD/data/train.jsonl")
ANSWER_INSTRUCTION = re.compile(
    r"\n*\s*Answer with the option(?:'s)? letter.*$",
    flags=re.IGNORECASE | re.DOTALL,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=Path("data/vision_opd"))
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--max-source-rows",
        type=int,
        default=None,
        help="Use only the first N source rows before making the stratified split.",
    )
    parser.add_argument(
        "--extract-original-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Extract original_images.tar.gz when the unboxed images are absent.",
    )
    return parser.parse_args()


def clean_question(item: dict[str, Any]) -> str:
    extra_question = (item.get("extra_info") or {}).get("question")
    question = str(extra_question or item.get("problem", ""))
    question = question.replace("<image>", "").strip()
    question = question.replace(
        "Only focus on the objects inside the red bounding box in the image "
        "to answer this question.",
        "",
    )
    return ANSWER_INSTRUCTION.sub("", question).strip()


def _validate_archive_members(archive: Path) -> None:
    listing = subprocess.run(
        ["tar", "-tf", str(archive)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    for name in listing:
        member = PurePosixPath(name)
        if member.is_absolute() or ".." in member.parts:
            raise ValueError(f"Unsafe path in {archive}: {name}")


def ensure_original_images(rows: list[dict[str, Any]], source_root: Path) -> None:
    expected = [source_root / str(item["original_images"][0]) for item in rows]
    if all(path.is_file() for path in expected):
        return
    archive = source_root / "original_images" / "original_images.tar.gz"
    if not archive.is_file():
        raise FileNotFoundError(
            f"Missing unboxed images and archive: {archive}. "
            "The red-box overlay images must not be used as Student inputs."
        )
    _validate_archive_members(archive)
    print(f"Extracting leakage-free original images from {archive} ...")
    subprocess.run(
        ["tar", "-xf", str(archive), "-C", str(archive.parent)],
        check=True,
    )
    missing = [path for path in expected if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Archive extraction left {len(missing)} images missing")


def stratified_split_indices(
    rows: list[dict[str, Any]], test_ratio: float, random_state: int
) -> tuple[list[int], list[int]]:
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(rows):
        answer = str(item.get("answer", "")).strip().upper()
        if answer not in {"A", "B", "C", "D"}:
            raise ValueError(f"Unsupported answer {answer!r} at source row {index}")
        groups[answer].append(index)

    # Match common holdout-split behavior and keep the 5,928-row train split
    # divisible by the historical eight-rollout group (verl may drop an
    # incomplete prompt batch for a particular launcher batch size).
    target_test_size = math.ceil(len(rows) * test_ratio)
    allocation = {label: math.floor(len(indices) * test_ratio) for label, indices in groups.items()}
    remaining = target_test_size - sum(allocation.values())
    fractional_order = sorted(
        groups,
        key=lambda label: (len(groups[label]) * test_ratio - allocation[label], label),
        reverse=True,
    )
    for label in fractional_order[:remaining]:
        allocation[label] += 1

    rng = random.Random(random_state)
    test_indices: list[int] = []
    train_indices: list[int] = []
    for label in sorted(groups):
        indices = groups[label][:]
        rng.shuffle(indices)
        split_at = allocation[label]
        test_indices.extend(indices[:split_at])
        train_indices.extend(indices[split_at:])
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    return train_indices, test_indices


def build_record(
    item: dict[str, Any], source_root: Path, source_index: int, split: str
) -> dict[str, Any]:
    question = clean_question(item)
    answer = str(item["answer"]).strip().upper()
    image_path = (source_root / str(item["original_images"][0])).resolve()
    oracle_crop_path = (source_root / str(item["teacher_images"][0])).resolve()
    oracle_overlay_path = (source_root / str(item["images"][0])).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    prompt = (
        "<image>\n"
        f"{question}\n"
        "Inspect the image carefully and reason only from visible evidence. State one concise, "
        "evidence-first rationale without <think> tags or hidden-chain-of-thought. Do not hedge, "
        "revise, or mention this instruction. On the final line write exactly `FINAL: X`, where X "
        "is the direct final answer. If the question explicitly uses labeled answer choices, X is "
        "the selected choice label."
    )
    return {
        "data_source": "vision_opd_6k_groove",
        "prompt": [{"role": "user", "content": prompt}],
        "images": [{"path": str(image_path)}],
        "ability": "visual_question_answering",
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": {
            "answer": answer,
            "question": question,
            "image_path": str(image_path),
            "question_id": f"vision-opd-{source_index:06d}",
            "split": split,
            # Audit-only Oracle metadata. The Student, Analyzer, and online
            # grounder never receive these fields.
            "oracle_bbox": [int(value) for value in item["bbox"]],
            "oracle_crop_path": str(oracle_crop_path),
            "oracle_overlay_path": str(oracle_overlay_path),
        },
    }


def answer_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(record["extra_info"]["answer"] for record in records)
    return {label: counts[label] for label in "ABCD"}


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    source_root = source.parent
    output_dir = args.output_dir.resolve()
    all_rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]
    if args.max_source_rows is not None:
        if args.max_source_rows <= 0:
            raise ValueError("--max-source-rows must be positive")
        if args.max_source_rows > len(all_rows):
            raise ValueError(
                f"--max-source-rows={args.max_source_rows} exceeds source size {len(all_rows)}"
            )
        rows = all_rows[: args.max_source_rows]
    else:
        rows = all_rows
    if args.extract_original_images:
        ensure_original_images(rows, source_root)

    train_indices, test_indices = stratified_split_indices(rows, args.test_ratio, args.random_state)
    if set(train_indices) & set(test_indices):
        raise RuntimeError("Train/test split overlap detected")
    if len(train_indices) + len(test_indices) != len(rows):
        raise RuntimeError("Train/test split does not cover every source row")

    train_records = [build_record(rows[index], source_root, index, "train") for index in train_indices]
    test_records = [build_record(rows[index], source_root, index, "test") for index in test_indices]
    train_images = {record["extra_info"]["image_path"] for record in train_records}
    test_images = {record["extra_info"]["image_path"] for record in test_records}
    if train_images & test_images:
        raise RuntimeError("Original-image leakage across train and test")

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.parquet"
    test_path = output_dir / "test.parquet"
    Dataset.from_list(train_records).to_parquet(str(train_path))
    Dataset.from_list(test_records).to_parquet(str(test_path))
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = {
        "source": str(source),
        "source_sha256": source_sha256,
        "source_rows": len(all_rows),
        "selected_source_rows": len(rows),
        "max_source_rows": args.max_source_rows,
        "random_state": args.random_state,
        "test_ratio": args.test_ratio,
        "student_image_policy": "unboxed_original_images_only",
        "oracle_metadata_visible_to_model": False,
        "train": {"rows": len(train_records), "answers": answer_counts(train_records)},
        "test": {"rows": len(test_records), "answers": answer_counts(test_records)},
    }
    (output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"Wrote {train_path} and {test_path}")


if __name__ == "__main__":
    main()
