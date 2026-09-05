#!/usr/bin/env python3
"""Run the updated English Analyzer end-to-end on the selected formal rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from groove.analyzer import OpenAIAnalyzerConfig, OpenAICompatibleAnalyzer
from groove.schemas import GroupRollout, Rollout
from recheck_english_grounding import (
    SAMPLES,
    annotate,
    crop_and_resize,
    load_json,
    question_from_evidence,
    rollout_info,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROLLOUTS = ROOT / "outputs/rollouts-batch8-seq64-formal-v1"
DEFAULT_EVIDENCE = ROOT / "outputs/evidence-batch8-seq64-formal-v1"
DEFAULT_OUTPUT = ROOT / "outputs/grounding-recheck-auto-analyzer"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", type=Path, default=DEFAULT_ROLLOUTS)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-crop-side", type=int, default=1600)
    return parser.parse_args()


def load_group(rollouts_dir: Path, sample: dict) -> GroupRollout:
    step = int(sample["step"])
    evidence_uid = str(sample["evidence_uid"])
    raw_uid = evidence_uid.replace(f"step-{step:07d}-", "", 1)
    rows_all = [
        json.loads(line)
        for line in (rollouts_dir / f"{step}.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [row for row in rows_all if row.get("uid") == raw_uid]
    evidence = load_json(Path(sample["evidence_path"]))
    target_question = question_from_evidence(evidence)
    if not rows:
        # Early formal-v1 steps predate the per-group UID field.  Recover those
        # groups by their exact question text rather than silently dropping them.
        def clean_question(row: dict) -> str:
            value = str(row["input"])
            if value.startswith("user\n\n"):
                value = value[len("user\n\n") :]
            return value.split("\nInspect the image carefully", 1)[0].strip()

        rows = [row for row in rows_all if clean_question(row) == target_question]
    if not rows:
        raise ValueError(f"No rollout rows found for {evidence_uid}")
    question = str(rows[0]["input"])
    if question.startswith("user\n\n"):
        question = question[len("user\n\n") :]
    question = question.split("\nInspect the image carefully", 1)[0].strip()
    evidence_path = Path(sample["evidence_path"])
    image_path = Path(load_json(evidence_path)["original_image_path"]).resolve()
    return GroupRollout(
        uid=evidence_uid,
        question=question,
        image_path=image_path,
        rollouts=[
            Rollout(
                rollout_id=index,
                completion=str(row["output"]),
                predicted_label=row.get("predicted_label"),
                is_correct=bool(float(row.get("accuracy", row.get("score", 0.0))) > 0.5),
            )
            for index, row in enumerate(rows)
        ],
    )


def trace_detections(trace: list[dict]) -> list[dict]:
    detections = []
    for item in trace:
        if item.get("name") != "ground_image":
            continue
        result = item.get("result") or {}
        bbox = result.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        detections.append(
            {
                "query": str(item.get("arguments", {}).get("query", result.get("query", ""))),
                "score": float(result.get("score", 0.0)),
                "bbox": bbox,
                "found": bool(result.get("found", False)),
            }
        )
    return detections


def focus_detections(focus: object) -> list[dict]:
    """Convert only the promoted final regions into displayable detections."""
    return [
        {
            "query": region.query,
            "score": float(region.score),
            "bbox": list(region.expanded_box),
            "found": True,
        }
        for region in focus.tool_regions
        if region.source == "grounding_dino"
    ]


def write_sample(
    output_dir: Path,
    ordinal: int,
    sample: dict,
    evidence: dict,
    group: GroupRollout,
    analyzer: OpenAICompatibleAnalyzer,
    focus: object,
    rollouts_dir: Path,
    max_crop_side: int,
) -> dict:
    sample_dir = output_dir / f"{ordinal:02d}-step-{int(sample['step']):04d}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(group.image_path) as loaded:
        image = loaded.convert("RGB")

    attempt_detections = trace_detections(analyzer.last_tool_trace)
    detections = focus_detections(focus)
    annotated_path = sample_dir / "original_with_auto_analyzer_boxes.jpg"
    annotate(image, detections).save(annotated_path, quality=92, optimize=True)
    crops = []
    for index, detection in enumerate(detections, start=1):
        crop_path = None
        if detection.get("found"):
            crop_path = sample_dir / f"auto-crop-{index:02d}.jpg"
            crop_and_resize(image, detection["bbox"], max_crop_side).save(
                crop_path, quality=92, optimize=True
            )
        row = dict(detection)
        row["crop_path"] = str(crop_path.resolve()) if crop_path else None
        bbox = detection["bbox"]
        row["crop_area_fraction"] = (
            (float(bbox[2]) - float(bbox[0])) * (float(bbox[3]) - float(bbox[1]))
            / (image.width * image.height)
        )
        crops.append(row)

    roll_info = rollout_info(rollouts_dir, int(sample["step"]), sample["evidence_uid"])
    if not roll_info.get("available"):
        roll_info = {
            "available": True,
            "uid": group.uid,
            "rollouts": len(group.rollouts),
            "correct": sum(item.is_correct for item in group.rollouts),
            "predicted_labels": [item.predicted_label or "" for item in group.rollouts],
            "ground_truth": next(
                (str(item.predicted_label) for item in group.rollouts if item.is_correct),
                None,
            ),
            "uid_source": "question_fallback",
        }
    record = {
        "ordinal": ordinal,
        "step": int(sample["step"]),
        "evidence_uid": sample["evidence_uid"],
        "question": group.question,
        "image_path": str(group.image_path),
        "rollout_info": roll_info,
        "historical_evidence_path": str(Path(sample["evidence_path"]).resolve()),
        "historical_focus": evidence.get("focus"),
        "focus": focus.model_dump(mode="json"),
        "tool_trace": analyzer.last_tool_trace,
        "attempt_detections": attempt_detections,
        "auto_detections": crops,
        "annotated_image": str(annotated_path.resolve()),
    }
    (sample_dir / "analysis.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        f"# Automatic English Analyzer recheck — Step {record['step']}",
        "",
        f"**Question**: {record['question']}",
        "",
        f"**Input image**: `{record['image_path']}`",
        "",
        f"**Original Rollout group**: `{record['rollout_info']}`",
        "",
        "## New Analyzer focus",
        "",
        "```json",
        json.dumps(record["focus"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Automatic tool trace",
        "",
        f"Tool calls: {len(record['tool_trace'])}; visual feedback attached: "
        f"{sum(bool(item.get('visual_feedback_attached')) for item in record['tool_trace'])}",
        "",
        "## Red-box result",
        "",
        f"![Automatic Analyzer boxes]({annotated_path.resolve()})",
        "",
    ]
    for index, crop in enumerate(crops, start=1):
        lines.extend(
            [
                f"### Tool call {index}: `{crop['query']}` (score {crop['score']:.3f}, area {crop['crop_area_fraction'] * 100:.2f}%)",
                "",
            ]
        )
        if crop.get("crop_path"):
            lines.extend([f"![Automatic Analyzer crop]({crop['crop_path']})", ""])
    (sample_dir / "review.md").write_text("\n".join(lines), encoding="utf-8")
    return record


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    analyzer = OpenAICompatibleAnalyzer(OpenAIAnalyzerConfig.from_env())
    records = []
    for ordinal, sample in enumerate(SAMPLES, start=1):
        sample = dict(sample)
        sample["evidence_path"] = args.evidence / sample["evidence_uid"] / "evidence.json"
        evidence = load_json(sample["evidence_path"])
        print(f"[{ordinal}/10] Step {sample['step']}: calling Analyzer", flush=True)
        try:
            group = load_group(args.rollouts, sample)
            focus = analyzer.analyze(group)
            record = write_sample(
                args.output_dir,
                ordinal,
                sample,
                evidence,
                group,
                analyzer,
                focus,
                args.rollouts,
                args.max_crop_side,
            )
            records.append(record)
            print(
                f"  focus={focus.tool_route}/{focus.crucial_evidence_type}, "
                f"calls={len(record['tool_trace'])}, regions={len(record['auto_detections'])}",
                flush=True,
            )
        except Exception as exc:
            error = {
                "ordinal": ordinal,
                "step": int(sample["step"]),
                "evidence_uid": sample["evidence_uid"],
                "error": f"{type(exc).__name__}: {exc}",
            }
            (args.output_dir / f"{ordinal:02d}-step-{int(sample['step']):04d}").mkdir(
                parents=True, exist_ok=True
            )
            (args.output_dir / f"{ordinal:02d}-step-{int(sample['step']):04d}" / "analysis.json").write_text(
                json.dumps(error, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            records.append(error)
            print(f"  ERROR: {error['error']}", flush=True)

    (args.output_dir / "review.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# Automatic English Analyzer recheck",
        "",
        "These records reuse the original formal-v1 Rollout groups and run the updated Analyzer with English prompts, GroundingDINO, and exact-crop visual feedback.",
        "",
    ]
    for record in records:
        sample_dir = args.output_dir / f"{int(record['ordinal']):02d}-step-{int(record['step']):04d}"
        lines.extend(
            [
                f"## Step {record['step']}",
                "",
                f"**Question**: {record.get('question', 'Analyzer failed before question was recorded')}",
                "",
                f"**Per-sample review**: [review.md]({(sample_dir / 'review.md').resolve()})",
                "",
            ]
        )
    (args.output_dir / "review.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(records)} automatic Analyzer records to {args.output_dir}")


if __name__ == "__main__":
    main()
