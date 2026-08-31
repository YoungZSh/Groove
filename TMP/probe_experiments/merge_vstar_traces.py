#!/usr/bin/env python3
"""Merge V*Bench JSONL shards, validate coverage, and write a summary."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected", type=int, default=191)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records: dict[int, dict[str, Any]] = {}
    for path in args.inputs:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                index = int(record["index"])
                previous = records.get(index)
                if previous is None or "error" in previous:
                    records[index] = record
                elif "error" in record:
                    continue
                elif previous.get("truncated") and not record.get("truncated"):
                    records[index] = record
                elif record.get("truncated") and not previous.get("truncated"):
                    continue
                elif previous.get("truncated") and record.get("truncated"):
                    if int(record.get("output_tokens", 0)) > int(
                        previous.get("output_tokens", 0)
                    ):
                        records[index] = record
                elif previous != record:
                    raise ValueError(f"Conflicting successful index {index} in {path}:{line_number}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for index in sorted(records):
            handle.write(json.dumps(records[index], ensure_ascii=False) + "\n")

    successful = [record for record in records.values() if "error" not in record]
    errors = [record for record in records.values() if "error" in record]
    scored = [record for record in successful if record.get("correct") is not None]
    correct = sum(bool(record["correct"]) for record in scored)
    missing = sorted(set(range(args.expected)) - records.keys())
    summary = {
        "expected": args.expected,
        "records": len(records),
        "successful": len(successful),
        "errors": len(errors),
        "missing_indices": missing,
        "scored": len(scored),
        "correct": correct,
        "accuracy": correct / len(scored) if scored else None,
        "truncated": sum(bool(record.get("truncated")) for record in successful),
        "by_category": {},
    }
    for category in sorted({record["category"] for record in scored}):
        category_records = [record for record in scored if record["category"] == category]
        category_correct = sum(bool(record["correct"]) for record in category_records)
        summary["by_category"][category] = {
            "count": len(category_records),
            "correct": category_correct,
            "accuracy": category_correct / len(category_records),
        }
    summary["prediction_counts"] = dict(
        Counter(record.get("predicted_label") for record in successful)
    )

    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"merged_output={args.output}")
    print(f"summary_output={summary_path}")
    return 0 if len(records) == args.expected and not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
