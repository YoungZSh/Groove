#!/usr/bin/env python3
"""Prepare the complete 191-question V*Bench validation set with original images."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from groove.response_prompt import REASONING_SYSTEM_PROMPT
from groove.vstar_bench import CHOICE_RE, DATA_SOURCE, question_text


DEFAULT_SOURCE = Path("/root/siton-tmp/yzs/datasets/vstar-bench/data/test-00000-of-00001.parquet")
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data/vstar_bench/validation.parquet"
EXPECTED_CATEGORIES = {"direct_attributes": 115, "relative_position": 76}


def build_records(rows: list[dict]) -> list[dict]:
    if len(rows) != 191 or len({str(row["question_id"]) for row in rows}) != 191:
        raise ValueError("V*Bench validation requires all 191 unique question IDs")
    if Counter(row["category"] for row in rows) != EXPECTED_CATEGORIES:
        raise ValueError("V*Bench validation category counts must be 115 and 76")
    records = []
    for row in rows:
        payload = row["image"].get("bytes")
        if not payload:
            raise ValueError("V*Bench validation requires original embedded image bytes")
        choices = dict(CHOICE_RE.findall(row["text"]))
        answer = str(row["label"]).strip().upper()
        if answer not in choices:
            raise ValueError("V*Bench reference label must occur in the answer choices")
        question = question_text(row["text"])
        records.append({
            "data_source": DATA_SOURCE,
            "prompt": [
                {"role": "system", "content": REASONING_SYSTEM_PROMPT},
                {"role": "user", "content": "<image>\n" + question},
            ],
            "images": [{"bytes": payload}],
            "ability": "visual_question_answering",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                "question_id": str(row["question_id"]),
                "question": question,
                "category": str(row["category"]),
                "choices": choices,
                "split": "validation",
                "image_sha256": hashlib.sha256(payload).hexdigest(),
            },
        })
    return records


def prepare_validation(source: Path, output: Path) -> dict:
    manifest_path = output.with_suffix(".manifest.json")
    if output.exists() or manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing validation artifacts: {output}")
    records = build_records(pq.read_table(source).to_pylist())
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        pq.write_table(pa.Table.from_pylist(records), handle)
    manifest = {
        "source": str(source.resolve()),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "validation_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "rows": len(records),
        "categories": EXPECTED_CATEGORIES,
        "image_policy": "original embedded bytes, unchanged source order",
        "scorer": "groove.vstar_bench.compute_validation_score",
        "system_prompt": REASONING_SYSTEM_PROMPT,
    }
    with manifest_path.open("x") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = prepare_validation(args.source, args.output)
    print(f"Prepared {manifest['rows']} V*Bench validation questions at {args.output}")


if __name__ == "__main__":
    main()
