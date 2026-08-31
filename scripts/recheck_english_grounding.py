#!/usr/bin/env python3
"""Re-ground selected failed formal rollouts with English DINO queries.

The script keeps the original question/image and rollout group, changes only the
GroundingDINO query to an English noun phrase, and writes red-box audit images plus
enlarged crops and a Markdown review index.
"""

from __future__ import annotations

import argparse
import base64
import json
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROLLOUTS = ROOT / "outputs/rollouts-batch8-seq64-formal-v1"
DEFAULT_EVIDENCE = ROOT / "outputs/evidence-batch8-seq64-formal-v1"
DEFAULT_OUTPUT = ROOT / "outputs/grounding-recheck-english"


SAMPLES = [
    {
        "step": 4,
        "evidence_uid": "step-0000004-30d2b9f2-018f-4180-b9b7-99d2f7f6d04d",
        "queries": ["tureen and tray set on carpet", "metal tureen", "metal tray"],
    },
    {
        "step": 20,
        "evidence_uid": "step-0000020-3ed23cd2-b22d-419f-adfd-0027a5df429f",
        "queries": ["hanging lantern under arch", "square hanging lantern", "lantern metal frame"],
    },
    {
        "step": 50,
        "evidence_uid": "step-0000050-d1091c1f-edd7-45f4-b51f-18e05d5fa621",
        "queries": ["red and white traffic cone on sidewalk", "traffic cone on sidewalk", "red white traffic cone"],
    },
    {
        "step": 55,
        "evidence_uid": "step-0000055-fb64ed1d-fa57-436c-b6d9-5cd51d1ef22c",
        "queries": ["small black rectangular panel on blue tram", "black panel on blue tram"],
    },
    {
        "step": 81,
        "evidence_uid": "step-0000081-64c1a444-6e6a-4d82-8558-cb5491b36860",
        "queries": ["small religious poster on red wall", "white religious poster on red wall"],
    },
    {
        "step": 84,
        "evidence_uid": "step-0000084-2804b243-0feb-46d4-a443-3298df2fbe84",
        "queries": ["large snow sculpture behind the bed", "snow sculpture in ice cave"],
    },
    {
        "step": 92,
        "evidence_uid": "step-0000092-2a2dcddf-f389-4808-bbe9-3f3ed2c096a1",
        "queries": ["animal in front passenger seat of white convertible", "animal in car passenger seat"],
    },
    {
        "step": 100,
        "evidence_uid": "step-0000100-c0111de1-5dd3-4351-a6df-9c1d53652457",
        "queries": ["person standing by bridge railing", "person wearing jacket by bridge"],
    },
    {
        "step": 113,
        "evidence_uid": "step-0000113-38188716-9834-435a-8bda-6c1e83596d47",
        "queries": ["person hair in lower right corner", "head hair in lower right corner"],
    },
    {
        "step": 429,
        "evidence_uid": "step-0000429-37468e22-842a-4de3-8d18-06c79308a3ef",
        "queries": ["person riding a bicycle", "cyclist wearing red shirt"],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", type=Path, default=DEFAULT_ROLLOUTS)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dino-url", default="http://127.0.0.1:8011")
    parser.add_argument("--context-margin", type=float, default=0.12)
    parser.add_argument("--box-threshold", type=float, default=0.15)
    parser.add_argument("--text-threshold", type=float, default=0.15)
    parser.add_argument("--max-crop-side", type=int, default=1600)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def dino_call(url: str, image_path: Path, query: str, args: argparse.Namespace) -> dict:
    payload = {
        "image_base64": base64.b64encode(image_path.read_bytes()).decode("ascii"),
        "query": query,
        "context_margin": args.context_margin,
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.loads(response.read().decode("utf-8"))


def crop_and_resize(image: Image.Image, bbox: list[float], max_side: int) -> Image.Image:
    width, height = image.size
    x1, y1, x2, y2 = [float(value) for value in bbox]
    box = (
        max(0, min(width, round(x1))),
        max(0, min(height, round(y1))),
        max(0, min(width, round(x2))),
        max(0, min(height, round(y2))),
    )
    if box[0] >= box[2] or box[1] >= box[3]:
        raise ValueError(f"Invalid bbox {bbox} for image size {image.size}")
    crop = image.crop(box)
    scale = min(1.0, max_side / max(crop.size))
    if scale < 1.0:
        crop = crop.resize(
            (max(1, round(crop.width * scale)), max(1, round(crop.height * scale))),
            Image.Resampling.LANCZOS,
        )
    elif min(crop.size) < 768:
        scale = 768 / max(min(crop.size), 1)
        scale = min(scale, max_side / max(crop.size))
        crop = crop.resize(
            (max(1, round(crop.width * scale)), max(1, round(crop.height * scale))),
            Image.Resampling.LANCZOS,
        )
    return crop


def annotate(image: Image.Image, detections: list[dict]) -> Image.Image:
    result = image.copy()
    draw = ImageDraw.Draw(result)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", max(18, round(min(image.size) / 80)))
    except OSError:
        font = ImageFont.load_default()
    line_width = max(4, round(min(image.size) / 350))
    for index, detection in enumerate(detections, start=1):
        bbox = detection.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [round(float(value)) for value in bbox]
        draw.rectangle((x1, y1, x2, y2), outline=(230, 20, 20), width=line_width)
        label = f"Q{index} {detection.get('query', '')} ({float(detection.get('score', 0.0)):.2f})"
        text_box = draw.textbbox((x1, y1), label, font=font)
        draw.rectangle(text_box, fill=(230, 20, 20))
        draw.text((x1, y1), label, fill=(255, 255, 255), font=font)
    return result


def question_from_evidence(evidence: dict) -> str:
    prompt = evidence.get("teacher_prompt") or []
    if prompt and isinstance(prompt[0], dict):
        content = str(prompt[0].get("content", ""))
        return content.replace("<image>\n", "", 1).split("\n\nHindsight visual focus", 1)[0].strip()
    return "Unknown question"


def rollout_info(rollouts_dir: Path, step: int, evidence_uid: str) -> dict:
    path = rollouts_dir / f"{step}.jsonl"
    if not path.is_file():
        return {"available": False}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    raw_uid = evidence_uid.replace(f"step-{step:07d}-", "", 1)
    group = [row for row in rows if row.get("uid") == raw_uid]
    if not group:
        return {"available": False, "uid": raw_uid}
    return {
        "available": True,
        "uid": raw_uid,
        "rollouts": len(group),
        "correct": sum(float(row.get("answer_reward", 0.0)) > 0.5 for row in group),
        "predicted_labels": [row.get("predicted_label", "") for row in group],
        "ground_truth": group[0].get("ground_truth_label"),
    }


def old_regions(evidence: dict) -> list[dict]:
    focus = evidence.get("focus") or {}
    regions = focus.get("tool_regions") or []
    return [
        {
            "query": region.get("query", ""),
            "score": region.get("score", 0.0),
            "bbox": region.get("expanded_box"),
            "source": region.get("source", ""),
        }
        for region in regions
    ]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict] = []
    for ordinal, sample in enumerate(SAMPLES, start=1):
        evidence_path = args.evidence / sample["evidence_uid"] / "evidence.json"
        evidence = load_json(evidence_path)
        image_path = Path(evidence["original_image_path"])
        sample_dir = args.output_dir / f"{ordinal:02d}-step-{sample['step']:04d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        with Image.open(image_path) as loaded:
            image = loaded.convert("RGB")
        detections = []
        for query in sample["queries"]:
            detection = dino_call(args.dino_url, image_path, query, args)
            detection = dict(detection)
            detection["query"] = query
            detections.append(detection)
        annotated = annotate(image, detections)
        annotated_path = sample_dir / "original_with_english_boxes.jpg"
        annotated.save(annotated_path, quality=92, optimize=True)

        crop_rows = []
        for crop_index, detection in enumerate(detections, start=1):
            bbox = detection.get("bbox")
            crop_path = None
            if isinstance(bbox, list) and len(bbox) == 4 and detection.get("found", False):
                crop = crop_and_resize(image, bbox, args.max_crop_side)
                crop_path = sample_dir / f"english-crop-{crop_index:02d}.jpg"
                crop.save(crop_path, quality=92, optimize=True)
            row = dict(detection)
            row["crop_path"] = str(crop_path.resolve()) if crop_path else None
            row["crop_area_fraction"] = (
                ((float(bbox[2]) - float(bbox[0])) * (float(bbox[3]) - float(bbox[1])))
                / (image.width * image.height)
                if isinstance(bbox, list) and len(bbox) == 4
                else None
            )
            crop_rows.append(row)

        record = {
            "ordinal": ordinal,
            "step": sample["step"],
            "evidence_uid": sample["evidence_uid"],
            "question": question_from_evidence(evidence),
            "image_path": str(image_path.resolve()),
            "rollout_info": rollout_info(args.rollouts, sample["step"], sample["evidence_uid"]),
            "focus": evidence.get("focus"),
            "old_regions": old_regions(evidence),
            "english_detections": crop_rows,
            "annotated_image": str(annotated_path.resolve()),
        }
        (sample_dir / "result.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        lines = [
            f"# English Grounding recheck — Step {sample['step']}",
            "",
            f"**Question**: {record['question']}",
            "",
            f"**Input image**: `{record['image_path']}`",
            "",
            f"**Original Rollout group**: `{record['rollout_info']}`",
            "",
            "## New red-box result",
            "",
            f"![Original image with English Grounding boxes]({annotated_path.resolve()})",
            "",
            "## New English crops",
            "",
        ]
        for crop_index, detection in enumerate(crop_rows, start=1):
            if detection.get("crop_path"):
                lines.extend(
                    [
                        f"### Q{crop_index}: `{detection['query']}` (score {float(detection.get('score', 0.0)):.3f})",
                        "",
                        f"![English crop {crop_index}]({detection['crop_path']})",
                        "",
                    ]
                )
        (sample_dir / "review.md").write_text("\n".join(lines), encoding="utf-8")
        index_rows.append(record)

    index_path = args.output_dir / "review.json"
    index_path.write_text(json.dumps(index_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    report = [
        "# English Grounding recheck",
        "",
        "These samples reuse the original formal-v1 rollout groups and images. Only the GroundingDINO queries were rewritten as English noun phrases.",
        "",
    ]
    for record in index_rows:
        report.extend(
            [
                f"## {record['ordinal']:02d}. Step {record['step']}",
                "",
                f"**Question**: {record['question']}",
                "",
                f"**Rollout group**: `{record['rollout_info']}`",
                "",
                f"**Input image**: [{Path(record['image_path']).name}]({record['image_path']})",
                "",
                f"**Red-box result**: [original_with_english_boxes.jpg]({record['annotated_image']})",
                "",
                "| Query | Score | Crop area | Crop |",
                "|---|---:|---:|---|",
            ]
        )
        for detection in record["english_detections"]:
            crop_link = (
                f"[{Path(detection['crop_path']).name}]({detection['crop_path']})"
                if detection.get("crop_path")
                else "not found"
            )
            area = detection.get("crop_area_fraction")
            report.append(
                f"| `{detection['query']}` | {float(detection.get('score', 0.0)):.3f} | "
                f"{area * 100:.1f}% | {crop_link} |"
                if area is not None
                else f"| `{detection['query']}` | {float(detection.get('score', 0.0)):.3f} | — | {crop_link} |"
            )
        sample_review_path = (
            args.output_dir / f"{record['ordinal']:02d}-step-{record['step']:04d}" / "review.md"
        ).resolve()
        report.extend(["", f"Full per-sample review: [review.md]({sample_review_path})", ""])
    (args.output_dir / "review.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote {len(index_rows)} English Grounding rechecks to {args.output_dir}")


if __name__ == "__main__":
    main()
