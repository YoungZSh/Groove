#!/usr/bin/env python3
"""Adapt Vision-OPD questions to unboxed Student inputs and semantic GRPO rewards."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import sys

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from groove.response_prompt import REASONING_SYSTEM_PROMPT
from groove.vstar_bench import SEMANTIC_ANSWER_INSTRUCTION, question_without_response_format


BOX_HINT = (
    "Only focus on the objects inside the red bounding box in the image "
    "to answer this question."
)
CHOICE = re.compile(r"^([A-D])\.\s*(.+)$", re.MULTILINE)
ANNOTATION_REFERENCE = re.compile(
    r"\bred bounding box\b|\b(?:highlighted|marked|boxed)\s+"
    r"(?:rectangular\s+)?(?:area|region|object)\b", re.IGNORECASE
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_question(item: dict) -> tuple[str, dict[str, str], str]:
    raw = (item.get("extra_info") or {}).get("question") or item.get("problem", "")
    text = question_without_response_format(str(raw).replace("<image>", "").replace(BOX_HINT, ""))
    matches = list(CHOICE.finditer(text))
    if len(matches) != 4 or [match[1] for match in matches] != list("ABCD"):
        raise ValueError("Expected exactly four ordered, nonempty A-D choices")
    stem = text[:matches[0].start()].strip()
    choices = {match[1]: match[2].strip() for match in matches}
    # Reject unparsed lines instead of quietly dropping part of a question.
    suffix = text[matches[0].start():]
    if CHOICE.sub("", suffix).strip() or not stem:
        raise ValueError("Unparsed question or choice text")
    answer = str(item["answer"]).strip().upper()
    if answer not in choices:
        raise ValueError("Reference answer must identify one of the choices")
    extra_answer = (item.get("extra_info") or {}).get("answer")
    if extra_answer is not None and str(extra_answer).strip().upper() != answer:
        raise ValueError("Conflicting reference labels")
    question = stem + "\n\n" + "\n".join(f"({key}) {value}" for key, value in choices.items())
    return question, choices, answer


def original_path(item: dict, source_root: Path) -> Path:
    images = item.get("original_images")
    if not isinstance(images, list) or len(images) != 1:
        raise ValueError("Exactly one unboxed original image is required")
    relative = Path(images[0])
    if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "original_images":
        raise ValueError("Student images must come from original_images")
    image = (source_root / relative).resolve()
    if not image.is_relative_to((source_root / "original_images").resolve()) or not image.is_file():
        raise ValueError("Missing or unsafe original image path")
    return image


def build_record(item: dict, source_root: Path, source_index: int, official_index: int) -> dict:
    question, choices, answer = parse_question(item)
    if ANNOTATION_REFERENCE.search(question.split("\n\n", 1)[0]):
        raise ValueError("Question still depends on a visual annotation")
    image = original_path(item, source_root)
    return {
        "data_source": "vision_opd_6k_grpo",
        "prompt": [
            {"role": "system", "content": REASONING_SYSTEM_PROMPT},
            {"role": "user", "content": "<image>\n" + question + "\n" + SEMANTIC_ANSWER_INSTRUCTION},
        ],
        "images": [{"path": str(image)}],
        "ability": "visual_question_answering",
        "reward_model": {"style": "rule", "ground_truth": f"({answer}) {choices[answer]}"},
        "extra_info": {
            "question": question, "choices": choices, "answer": answer,
            "image_path": str(image), "question_id": f"vision-opd-{official_index:06d}",
            "source_index": source_index, "official_source_index": official_index, "split": "train",
        },
    }


def audit_exclusions(audit_path: Path | None, source_sha256: str, rows: list[dict]) -> dict[int, dict]:
    if audit_path is None:
        return {}
    audit = json.loads(audit_path.read_text())
    if audit["method"]["dataset_sha256"] != source_sha256:
        raise ValueError("Audit source hash does not match this dataset")
    excluded = {}
    seen = set()
    for record in audit["records"]:
        index = record["filtered_row"]
        if type(index) is not int or not 0 <= index < len(rows) or index in seen:
            raise ValueError("Invalid or duplicate audit source index")
        seen.add(index)
        if Path(record["original_image"]).name != Path(rows[index]["original_images"][0]).name:
            raise ValueError("Audit original image does not match its source row")
        reasons = []
        if record["review_category"] == "confirmed_defect":
            reasons.append("audit_confirmed_defect")
        if record["unboxed_use"] == "rewrite_required":
            reasons.append("audit_unboxed_rewrite_required")
        if reasons:
            excluded[index] = {"reasons": reasons, "audit_id": record["audit_id"],
                               "audit_note": record["audit_notes"]}
    return excluded


def prepare(source: Path, output: Path, audit: Path | None = None,
            sample_size: int | None = None, seed: int = 20260904) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing data directory: {output}")
    source = source.resolve()
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    source_sha = digest(source)
    official_indices = list(range(len(rows)))
    source_manifest = source.with_name("manifest.json")
    if source_manifest.exists():
        metadata = json.loads(source_manifest.read_text())
        if metadata.get("filtered_jsonl_sha256") != source_sha:
            raise ValueError("Source manifest hash mismatch")
        official_indices = metadata["kept_source_indices"]
        if (len(official_indices) != len(rows) or len(set(official_indices)) != len(rows)
                or any(type(index) is not int or index < 0 for index in official_indices)):
            raise ValueError("Invalid official source indices")
    excluded = audit_exclusions(audit, source_sha, rows)
    records, lineage = [], []
    for index, item in enumerate(rows):
        question, _, _ = parse_question(item)
        if ANNOTATION_REFERENCE.search(question.split("\n\n", 1)[0]):
            excluded.setdefault(index, {"reasons": []})["reasons"].append("explicit_annotation_reference")
        if index in excluded:
            continue
        records.append(build_record(item, source.parent, index, official_indices[index]))
        # Oracle fields stay in this separate audit file, outside the training parquet.
        lineage.append({"source_index": index, "official_source_index": official_indices[index],
                        "oracle_bbox": item.get("bbox"), "oracle_images": item.get("images"),
                        "oracle_teacher_images": item.get("teacher_images")})
    if not records:
        raise ValueError("No training records remain")
    image_paths = [record["extra_info"]["image_path"] for record in records]
    if len(set(image_paths)) != len(image_paths):
        raise ValueError("Duplicate original image paths need an explicit sampling policy")
    eligible_rows = len(records)
    if sample_size is not None:
        if not 0 < sample_size <= eligible_rows:
            raise ValueError("Sample size must be positive and at most the eligible row count")
        selected = random.Random(seed).sample(range(eligible_rows), sample_size)
        records = [records[index] for index in selected]
        lineage = [lineage[index] for index in selected]
    output.mkdir(parents=True, exist_ok=False)
    with (output / "train.parquet").open("xb") as handle:
        pq.write_table(pa.Table.from_pylist(records), handle)
    for name, entries in (
        ("lineage.jsonl", lineage),
        ("excluded.jsonl", [{"source_index": index, "official_source_index": official_indices[index],
                              **record} for index, record in sorted(excluded.items())]),
    ):
        with (output / name).open("x") as handle:
            for entry in entries:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    manifest = {
        "source": str(source), "source_sha256": source_sha, "source_rows": len(rows),
        "rows": len(records), "eligible_rows": eligible_rows, "excluded_rows": len(excluded),
        "sample_size": sample_size, "sample_seed": seed if sample_size is not None else None,
        "sampling": "uniform without replacement" if sample_size is not None else "all eligible rows",
        "selected_source_indices": [record["extra_info"]["source_index"] for record in records],
        "exclusion_reason_counts": dict(Counter(reason for value in excluded.values() for reason in value["reasons"])),
        "audit": str(audit.resolve()) if audit else None, "audit_sha256": digest(audit) if audit else None,
        "train_sha256": digest(output / "train.parquet"),
        "row_order": ("seeded random sample order" if sample_size is not None else "source order")
                     + "; launcher performs seeded shuffle",
        "image_policy": "unboxed original_images paths; original image bytes unchanged",
        "oracle_metadata": "separate lineage.jsonl only; absent from training parquet",
        "reference_policy": "original answer label plus full option text; labels unchanged",
        "response_format": "reasoning_answer", "validation": "external unchanged V*Bench parquet",
        "quality_limit": "Only known audit defects, required target rewrites and explicit annotation references excluded; remaining rows are not fully audited",
    }
    with (output / "manifest.json").open("x") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()
    manifest = prepare(args.source, args.output_dir, args.audit, args.sample_size, args.seed)
    print(json.dumps({key: value for key, value in manifest.items() if key != "selected_source_indices"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
