#!/usr/bin/env python3
"""No-training probes for sparse interleaved visual-evidence OPD.

Probe A measures whether a same-model Teacher given an inserted evidence crop
produces a selective, directionally useful and teachable distributional signal
relative to the original-image-only Student context.

Probe B evaluates CoFFT-style attention + sliding-window localization against
the official V*Bench target boxes.  The strict paper formula is reported next
to a denominator-clipped log-ratio variant because the former can be dominated
by nearly-zero descriptive-attention denominators.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from analyze_vstar_attention import (
    capture_full_attention,
    find_subsequence,
    heat_overlay,
    load_trace_records,
    prepare_descriptive_inputs,
    prepare_teacher_forced_inputs,
    relative_attention,
    sentence_char_spans,
)


DEFAULT_MODEL = Path("/root/siton-tmp/yzs/ckpts/Qwen3.5-4B")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = Path(
    "/root/siton-tmp/yzs/datasets/vstar-bench/data/test-00000-of-00001.parquet"
)
DEFAULT_TRACES = Path(
    PROJECT_ROOT / "outputs/qwen3.5-4b-vstar/traces.jsonl"
)
DEFAULT_ANNOTATIONS = Path(
    "/root/siton-tmp/yzs/GLaQ/benchmark_zoomeye_multibench/"
    "official_annotations/annotation_vstar.json"
)
DEFAULT_OUTPUT = Path(
    PROJECT_ROOT / "outputs/interleaved-opd-probe"
)
IMAGE_MARKER = "<|vision_start|><|image_pad|><|vision_end|>"
METHODS = ("absolute", "relative_strict", "relative_stable")
CONDITIONS = (
    "none",
    "blank",
    "random",
    "absolute",
    "relative_strict",
    "relative_stable",
    "oracle",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--traces", type=Path, default=DEFAULT_TRACES)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pilot-per-stratum", type=int, default=10)
    parser.add_argument("--indices", type=int, nargs="+")
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--random-window-draws", type=int, default=200)
    parser.add_argument("--summarize-only", action="store_true")
    return parser.parse_args()


def normalized_question(value: str) -> str:
    return re.sub(r"\s+", " ", value.splitlines()[0].strip()).rstrip(".").lower()


def load_annotations(path: Path) -> dict[str, dict[str, Any]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    return {normalized_question(record["question"]): record for record in records}


def select_indices(
    traces: dict[int, dict[str, Any]],
    explicit: list[int] | None,
    per_stratum: int,
    seed: int,
) -> list[int]:
    if explicit:
        return sorted(set(explicit))
    strata: dict[tuple[str, bool], list[int]] = defaultdict(list)
    for index, record in traces.items():
        strata[(record["category"], bool(record["correct"]))].append(index)
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for key in sorted(strata):
        candidates = np.asarray(sorted(strata[key]), dtype=np.int64)
        count = min(per_stratum, len(candidates))
        selected.extend(int(value) for value in rng.choice(candidates, count, replace=False))
    return sorted(selected)


def prepare_retokenized_teacher_inputs(
    processor: Any,
    image: Image.Image,
    record: dict[str, Any],
    device: str,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], list[int], tuple[int, int]]:
    """Fallback for rare decoded completions that do not round-trip generated IDs."""
    batch = processor(
        text=[record["prompt"] + record["completion"]],
        images=[image],
        return_tensors="pt",
    )
    input_ids = batch["input_ids"][0].tolist()
    specs: list[dict[str, Any]] = []
    for sentence_index, (start, end, text) in enumerate(sentence_char_spans(record["completion"]), start=1):
        sentence_ids = processor.tokenizer(text, add_special_tokens=False)["input_ids"]
        found_start = None
        dropped = 0
        # Leading whitespace at a sentence boundary can change only the first
        # token.  Prefer the full span, then progressively drop at most two.
        for dropped in range(min(3, len(sentence_ids))):
            try:
                found_start = find_subsequence(input_ids, sentence_ids[dropped:])
                break
            except ValueError:
                continue
        if found_start is None:
            raise ValueError(f"Could not align sentence {sentence_index} for index {record['index']}")
        specs.append(
            {
                "sentence_index": sentence_index,
                "text": text,
                "char_start": start,
                "char_end": end,
                "query_positions": [
                    found_start + offset - 1 for offset in range(len(sentence_ids) - dropped)
                ],
            }
        )
    image_positions = torch.where(batch["mm_token_type_ids"][0] == 1)[0].tolist()
    grid_t, grid_h, grid_w = [int(value) for value in batch["image_grid_thw"][0].tolist()]
    if grid_t != 1:
        raise ValueError("The probe expects a still image")
    grid = (grid_h // 2, grid_w // 2)
    if len(image_positions) != math.prod(grid):
        raise ValueError("Retokenized image grid mismatch")
    return (
        {key: value.to(device) for key, value in batch.items()},
        specs,
        image_positions,
        grid,
    )


def stable_relative_attention(
    task_attention: np.ndarray,
    descriptive_attention: np.ndarray,
    grid: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, float]:
    """Denominator-clipped, spatially-smoothed log-ratio attention.

    The strict CoFFT outer softmax exponentiates an already large ratio.  Here
    the denominator is floored at its positive 25th percentile and the ratio is
    linearly normalized (equivalent to softmax(log(ratio))).
    """
    task = task_attention.astype(np.float64)
    baseline = descriptive_attention.astype(np.float64)
    positive = baseline[baseline > 0]
    floor = float(np.quantile(positive, 0.25)) if positive.size else 1e-10
    ratio = (task + 1e-10) / (np.maximum(baseline, floor) + 1e-10)
    tensor = torch.from_numpy(ratio.reshape(1, 1, *grid)).float()
    smoothed = F.avg_pool2d(tensor, kernel_size=3, stride=1, padding=1)[0, 0]
    ratio = smoothed.numpy().reshape(-1).astype(np.float64)
    total = float(ratio.sum())
    weights = ratio / total if total > 0 else np.full_like(ratio, 1 / ratio.size)
    return ratio.astype(np.float32), weights.astype(np.float32), floor


def probability_map(values: np.ndarray) -> np.ndarray:
    result = np.maximum(values.astype(np.float64), 0)
    total = float(result.sum())
    if total <= 0:
        return np.full(result.shape, 1 / result.size, dtype=np.float32)
    return (result / total).astype(np.float32)


def crop_score_map(
    question_map: np.ndarray,
    prefix_map: np.ndarray | None,
    sentence_map: np.ndarray,
    alpha: float = 0.3,
) -> np.ndarray:
    context = question_map if prefix_map is None else np.maximum(question_map - alpha * prefix_map, 0)
    return probability_map(0.5 * probability_map(context) + 0.5 * probability_map(sentence_map))


def possible_starts(total: int, width: int, stride: int) -> list[int]:
    starts = list(range(0, max(total - width + 1, 1), stride))
    last = max(total - width, 0)
    if not starts or starts[-1] != last:
        starts.append(last)
    return starts


def best_sliding_window(
    heat: np.ndarray,
    grid: tuple[int, int],
) -> tuple[int, int, int, int, float]:
    heat2d = heat.reshape(grid)
    integral = np.pad(heat2d.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    rows, cols = grid
    stride_r = max(1, round(rows * 0.10))
    stride_c = max(1, round(cols * 0.10))
    mass_total = float(heat2d.sum())
    if mass_total > 0:
        rr, cc = np.indices(grid)
        centroid_r = float((rr * heat2d).sum() / mass_total)
        centroid_c = float((cc * heat2d).sum() / mass_total)
    else:
        centroid_r, centroid_c = (rows - 1) / 2, (cols - 1) / 2
    best: tuple[tuple[float, float, float], tuple[int, int, int, int, float]] | None = None
    for fraction in np.arange(0.4, 0.91, 0.1):
        height = min(rows, max(1, round(rows * float(fraction))))
        width = min(cols, max(1, round(cols * float(fraction))))
        area = height * width
        for top in possible_starts(rows, height, stride_r):
            for left in possible_starts(cols, width, stride_c):
                bottom, right = top + height, left + width
                mass = float(
                    integral[bottom, right]
                    - integral[top, right]
                    - integral[bottom, left]
                    + integral[top, left]
                )
                density = mass / area
                distance = math.hypot(
                    (top + bottom - 1) / 2 - centroid_r,
                    (left + right - 1) / 2 - centroid_c,
                )
                # Density is the primary CoFFT criterion.  Centroid distance
                # resolves the common one-hot/tied-window case deterministically.
                key = (density, -distance, -area)
                value = (top, left, bottom, right, density)
                if best is None or key > best[0]:
                    best = (key, value)
    assert best is not None
    return best[1]


def grid_window_to_pixels(
    window: tuple[int, int, int, int, float],
    grid: tuple[int, int],
    image_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    top, left, bottom, right, _ = window
    rows, cols = grid
    width, height = image_size
    return (
        max(0, math.floor(left * width / cols)),
        max(0, math.floor(top * height / rows)),
        min(width, math.ceil(right * width / cols)),
        min(height, math.ceil(bottom * height / rows)),
    )


def annotation_boxes(annotation: dict[str, Any]) -> list[tuple[float, float, float, float]]:
    return [
        (float(x), float(y), float(x + width), float(y + height))
        for x, y, width, height in annotation["bbox"]
    ]


def intersection_area(a: Iterable[float], b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )


def union_box(boxes: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def localization_metrics(
    window: tuple[int, int, int, int],
    peak: tuple[float, float],
    boxes: list[tuple[float, float, float, float]],
    image_size: tuple[int, int],
) -> dict[str, float | bool]:
    coverages = []
    centers_inside = []
    for box in boxes:
        area = max((box[2] - box[0]) * (box[3] - box[1]), 1e-9)
        coverages.append(intersection_area(window, box) / area)
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        centers_inside.append(window[0] <= cx <= window[2] and window[1] <= cy <= window[3])
    target_union = union_box(boxes)
    intersection = intersection_area(window, target_union)
    window_area = max((window[2] - window[0]) * (window[3] - window[1]), 1e-9)
    union_area = max(
        window_area
        + (target_union[2] - target_union[0]) * (target_union[3] - target_union[1])
        - intersection,
        1e-9,
    )
    tx, ty = (target_union[0] + target_union[2]) / 2, (target_union[1] + target_union[3]) / 2
    wx, wy = (window[0] + window[2]) / 2, (window[1] + window[3]) / 2
    diagonal = math.hypot(*image_size)
    peak_hit = any(
        box[0] <= peak[0] <= box[2] and box[1] <= peak[1] <= box[3] for box in boxes
    )
    return {
        "all_target_centers_inside": bool(all(centers_inside)),
        "any_target_center_inside": bool(any(centers_inside)),
        "mean_bbox_coverage": float(np.mean(coverages)),
        "min_bbox_coverage": float(np.min(coverages)),
        "union_iou": float(intersection / union_area),
        "window_center_distance": float(math.hypot(wx - tx, wy - ty) / diagonal),
        "peak_in_target_bbox": bool(peak_hit),
    }


def random_window_baseline(
    window: tuple[int, int, int, int],
    boxes: list[tuple[float, float, float, float]],
    image_size: tuple[int, int],
    draws: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    width, height = image_size
    crop_w, crop_h = window[2] - window[0], window[3] - window[1]
    values: list[dict[str, float | bool]] = []
    for _ in range(draws):
        left = int(rng.integers(0, max(width - crop_w, 0) + 1))
        top = int(rng.integers(0, max(height - crop_h, 0) + 1))
        candidate = (left, top, left + crop_w, top + crop_h)
        values.append(localization_metrics(candidate, (-1, -1), boxes, image_size))
    return {
        "all_target_centers_inside": float(np.mean([v["all_target_centers_inside"] for v in values])),
        "any_target_center_inside": float(np.mean([v["any_target_center_inside"] for v in values])),
        "mean_bbox_coverage": float(np.mean([v["mean_bbox_coverage"] for v in values])),
        "min_bbox_coverage": float(np.mean([v["min_bbox_coverage"] for v in values])),
        "union_iou": float(np.mean([v["union_iou"] for v in values])),
        "window_center_distance": float(np.mean([v["window_center_distance"] for v in values])),
    }


def oracle_window(
    boxes: list[tuple[float, float, float, float]],
    image_size: tuple[int, int],
    expansion: float = 2.0,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = union_box(boxes)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    width = max(x2 - x1, image_size[0] * 0.08) * expansion
    height = max(y2 - y1, image_size[1] * 0.08) * expansion
    left = max(0, int(round(cx - width / 2)))
    top = max(0, int(round(cy - height / 2)))
    right = min(image_size[0], int(round(cx + width / 2)))
    bottom = min(image_size[1], int(round(cy + height / 2)))
    left = max(0, right - int(round(width)))
    top = max(0, bottom - int(round(height)))
    return left, top, right, bottom


def target_logits(
    model: Qwen3_5ForConditionalGeneration,
    processor: Any,
    original: Image.Image,
    prefix: str,
    target: str,
    evidence: Image.Image | None,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if evidence is None:
        text = prefix + target
        images = [original]
    else:
        text = prefix + IMAGE_MARKER + "\n" + target
        images = [original, evidence]
    batch = processor(text=[text], images=images, return_tensors="pt")
    full_target_ids = processor.tokenizer(target, add_special_tokens=False)["input_ids"]
    # The first word can be tokenized as either `Word` or `ĠWord` depending on
    # whether the original rollout boundary was a space while the inserted
    # image boundary was a newline.  Drop only that boundary-sensitive token;
    # all remaining target tokens then have identical IDs in both contexts.
    target_ids = full_target_ids[1:]
    if not target_ids:
        raise ValueError("Selected target span is too short for aligned OPD scoring")
    target_start = find_subsequence(batch["input_ids"][0].tolist(), target_ids)
    query_positions = torch.arange(
        target_start - 1,
        target_start + len(target_ids) - 1,
        dtype=torch.long,
        device=device,
    )
    batch = {key: value.to(device) for key, value in batch.items()}
    model.config.text_config._attn_implementation = "flash_attention_2"
    with torch.inference_mode():
        logits = model(
            **batch,
            use_cache=False,
            logits_to_keep=query_positions,
        ).logits[0].float().cpu()
    return logits, torch.tensor(target_ids, dtype=torch.long)


def option_probabilities(
    model: Qwen3_5ForConditionalGeneration,
    processor: Any,
    original: Image.Image,
    prefix: str,
    evidence: Image.Image | None,
    options: list[str],
    device: str,
) -> dict[str, float]:
    suffix = "The correct answer is ("
    if evidence is None:
        text = prefix + suffix
        images = [original]
    else:
        text = prefix + IMAGE_MARKER + "\n" + suffix
        images = [original, evidence]
    batch = processor(text=[text], images=images, return_tensors="pt")
    candidate_ids = []
    for option in options:
        ids = processor.tokenizer(option, add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            raise ValueError(f"Option {option!r} is not one token")
        candidate_ids.append(ids[0])
    batch = {key: value.to(device) for key, value in batch.items()}
    model.config.text_config._attn_implementation = "flash_attention_2"
    with torch.inference_mode():
        logits = model(**batch, use_cache=False, logits_to_keep=1).logits[0, -1].float()
    probs = torch.softmax(logits[candidate_ids], dim=0).cpu().tolist()
    return {option: float(prob) for option, prob in zip(options, probs)}


def distribution_metrics(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    target_ids: torch.Tensor,
    top_k: int,
) -> dict[str, Any]:
    student_logp = F.log_softmax(student_logits, dim=-1)
    teacher_logp = F.log_softmax(teacher_logits, dim=-1)
    student_p, teacher_p = student_logp.exp(), teacher_logp.exp()
    midpoint = 0.5 * (student_p + teacher_p)
    log_midpoint = torch.log(midpoint.clamp_min(1e-30))
    teacher_to_student = (teacher_p * (teacher_logp - student_logp)).sum(-1)
    student_to_teacher = (student_p * (student_logp - teacher_logp)).sum(-1)
    jsd = 0.5 * (
        (teacher_p * (teacher_logp - log_midpoint)).sum(-1)
        + (student_p * (student_logp - log_midpoint)).sum(-1)
    )
    student_top = student_p.topk(top_k, dim=-1).indices
    teacher_top = teacher_p.topk(top_k, dim=-1).indices
    overlap = (student_top[:, :, None] == teacher_top[:, None, :]).any(-1).float().mean(-1)
    teacher_on_student = teacher_p.gather(1, student_top).sum(-1)
    student_on_teacher = student_p.gather(1, teacher_top).sum(-1)
    token_rows = torch.arange(target_ids.numel())
    student_nll = -student_logp[token_rows, target_ids]
    teacher_nll = -teacher_logp[token_rows, target_ids]
    return {
        "teacher_to_student_kl": float(teacher_to_student.mean()),
        "student_to_teacher_kl": float(student_to_teacher.mean()),
        "jsd": float(jsd.mean()),
        "topk_overlap": float(overlap.mean()),
        "teacher_mass_on_student_topk": float(teacher_on_student.mean()),
        "student_mass_on_teacher_topk": float(student_on_teacher.mean()),
        "student_target_nll": float(student_nll.mean()),
        "teacher_target_nll": float(teacher_nll.mean()),
        "target_nll_delta": float((student_nll - teacher_nll).mean()),
        "token_teacher_to_student_kl": teacher_to_student.tolist(),
        "token_teacher_mass_on_student_topk": teacher_on_student.tolist(),
    }


def parse_option_letters(record: dict[str, Any]) -> list[str]:
    return re.findall(r"^\(([A-Z])\)\s", record["question"], flags=re.MULTILINE)


def deterministic_rng(seed: int, index: int, salt: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}:{index}:{salt}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def save_localization_panel(
    image: Image.Image,
    grid: tuple[int, int],
    heatmaps: dict[str, np.ndarray],
    windows: dict[str, tuple[int, int, int, int]],
    boxes: list[tuple[float, float, float, float]],
    output: Path,
) -> None:
    panels: list[Image.Image] = []
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 25)
    for method in METHODS:
        heat = heatmaps[method]
        ceiling = max(float(np.quantile(heat, 0.99)), 1e-12)
        visual = np.sqrt(np.clip(heat / ceiling, 0, 1)).astype(np.float32)
        panel = heat_overlay(image, visual, grid)
        draw = ImageDraw.Draw(panel)
        for box in boxes:
            draw.rectangle(tuple(round(v) for v in box), outline=(60, 255, 80), width=max(4, image.width // 500))
        draw.rectangle(windows[method], outline=(255, 225, 55), width=max(5, image.width // 400))
        draw.rectangle((0, 0, min(image.width, 390), 48), fill=(20, 20, 20))
        draw.text((12, 8), method, font=font, fill=(255, 255, 255))
        panel.thumbnail((850, 520), Image.Resampling.LANCZOS)
        panels.append(panel)
    width = max(panel.width for panel in panels)
    canvas = Image.new("RGB", (width, sum(panel.height for panel in panels)), (245, 245, 245))
    top = 0
    for panel in panels:
        canvas.paste(panel, ((width - panel.width) // 2, top))
        top += panel.height
    canvas.save(output, quality=90, optimize=True)


def analyze_one(
    model: Qwen3_5ForConditionalGeneration,
    processor: Any,
    row: dict[str, Any],
    trace: dict[str, Any],
    annotation: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.time()
    index = int(trace["index"])
    image = Image.open(BytesIO(row["image"]["bytes"])).convert("RGB")
    boxes = annotation_boxes(annotation)
    sample_dir = args.output_dir / "samples" / f"index-{index:03d}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    try:
        inputs, sentence_specs, _, image_positions, grid = prepare_teacher_forced_inputs(
            processor, image, trace, args.device
        )
        attention_input_mode = "recorded_token_ids"
    except ValueError as error:
        if "Completion token round-trip mismatch" not in str(error):
            raise
        inputs, sentence_specs, image_positions, grid = prepare_retokenized_teacher_inputs(
            processor, image, trace, args.device
        )
        attention_input_mode = "retokenized_completion_fallback"
    question_ids = processor.tokenizer(trace["question"], add_special_tokens=False)["input_ids"]
    question_start = find_subsequence(inputs["input_ids"][0].tolist(), question_ids)
    question_spec = {
        "sentence_index": -1,
        "text": trace["question"],
        "query_positions": [question_start + offset - 1 for offset in range(len(question_ids))],
    }
    task_specs = [question_spec, *sentence_specs]
    layer_maps, layer_masses = capture_full_attention(model, inputs, task_specs, image_positions)
    del inputs

    baseline_inputs, baseline_specs, baseline_positions, baseline_grid = prepare_descriptive_inputs(
        processor, image, args.device
    )
    if baseline_grid != grid or baseline_positions != image_positions:
        raise ValueError(f"Image grid mismatch for index {index}")
    baseline_maps, _ = capture_full_attention(
        model, baseline_inputs, baseline_specs, baseline_positions
    )
    del baseline_inputs
    descriptive = baseline_maps[0].mean(axis=0)

    raw_maps = {key: value.mean(axis=0) for key, value in layer_maps.items()}
    absolute_maps = {key: probability_map(value) for key, value in raw_maps.items()}
    strict_maps = {
        key: relative_attention(value, descriptive)[1] for key, value in raw_maps.items()
    }
    stable_values = {
        key: stable_relative_attention(value, descriptive, grid) for key, value in raw_maps.items()
    }
    stable_maps = {key: value[1] for key, value in stable_values.items()}
    denominator_floor = float(next(iter(stable_values.values()))[2])

    candidate_specs = [
        spec
        for spec in sentence_specs
        if len(spec["query_positions"]) >= 3
        and not re.fullmatch(r"(?:\*\*)?\(?[A-Z]\)?[^a-zA-Z]*", spec["text"].strip())
    ] or sentence_specs
    selected_spec = max(
        candidate_specs,
        key=lambda spec: float(layer_masses[spec["sentence_index"]].mean()),
    )
    selected_id = int(selected_spec["sentence_index"])

    prefix_ids = [spec["sentence_index"] for spec in sentence_specs if spec["char_end"] <= selected_spec["char_start"]]
    prefix_raw = None
    if prefix_ids:
        weights = np.asarray(
            [len(next(spec for spec in sentence_specs if spec["sentence_index"] == key)["query_positions"]) for key in prefix_ids],
            dtype=np.float64,
        )
        prefix_raw = np.average(np.stack([raw_maps[key] for key in prefix_ids]), axis=0, weights=weights)

    family_maps: dict[str, dict[int, np.ndarray]] = {
        "absolute": absolute_maps,
        "relative_strict": strict_maps,
        "relative_stable": stable_maps,
    }
    selected_heatmaps: dict[str, np.ndarray] = {}
    pixel_windows: dict[str, tuple[int, int, int, int]] = {}
    localization: dict[str, Any] = {}
    random_baselines: dict[str, Any] = {}
    for method, maps in family_maps.items():
        if prefix_raw is None:
            prefix_map = None
        elif method == "absolute":
            prefix_map = probability_map(prefix_raw)
        elif method == "relative_strict":
            prefix_map = relative_attention(prefix_raw, descriptive)[1]
        else:
            prefix_map = stable_relative_attention(prefix_raw, descriptive, grid)[1]
        heat = crop_score_map(maps[-1], prefix_map, maps[selected_id])
        selected_heatmaps[method] = heat
        grid_window = best_sliding_window(heat, grid)
        pixel_window = grid_window_to_pixels(grid_window, grid, image.size)
        pixel_windows[method] = pixel_window
        peak_index = int(np.argmax(heat))
        peak_row, peak_col = divmod(peak_index, grid[1])
        peak = (
            (peak_col + 0.5) * image.width / grid[1],
            (peak_row + 0.5) * image.height / grid[0],
        )
        localization[method] = {
            **localization_metrics(pixel_window, peak, boxes, image.size),
            "window": list(pixel_window),
            "window_fraction": float(
                (pixel_window[2] - pixel_window[0])
                * (pixel_window[3] - pixel_window[1])
                / (image.width * image.height)
            ),
            "peak": [float(peak[0]), float(peak[1])],
        }
        random_baselines[method] = random_window_baseline(
            pixel_window,
            boxes,
            image.size,
            args.random_window_draws,
            deterministic_rng(args.seed, index, f"random-baseline-{method}"),
        )

    oracle = oracle_window(boxes, image.size)
    stable_window = pixel_windows["relative_stable"]
    rng = deterministic_rng(args.seed, index, "random-evidence")
    crop_width, crop_height = stable_window[2] - stable_window[0], stable_window[3] - stable_window[1]
    random_left = int(rng.integers(0, max(image.width - crop_width, 0) + 1))
    random_top = int(rng.integers(0, max(image.height - crop_height, 0) + 1))
    random_window = (random_left, random_top, random_left + crop_width, random_top + crop_height)
    mean_color = tuple(int(value) for value in np.asarray(image).reshape(-1, 3).mean(axis=0))
    evidence: dict[str, Image.Image | None] = {
        "none": None,
        "blank": Image.new("RGB", (crop_width, crop_height), mean_color),
        "random": image.crop(random_window),
        "absolute": image.crop(pixel_windows["absolute"]),
        "relative_strict": image.crop(pixel_windows["relative_strict"]),
        "relative_stable": image.crop(stable_window),
        "oracle": image.crop(oracle),
    }
    for condition, crop in evidence.items():
        if crop is not None:
            crop.save(sample_dir / f"crop-{condition}.jpg", quality=93)
    save_localization_panel(
        image,
        grid,
        selected_heatmaps,
        pixel_windows,
        boxes,
        sample_dir / "localization.jpg",
    )

    target = selected_spec["text"]
    completion_prefix = trace["completion"][: selected_spec["char_start"]]
    context_prefix = trace["prompt"] + completion_prefix
    logits_by_condition: dict[str, torch.Tensor] = {}
    target_ids: torch.Tensor | None = None
    option_probs: dict[str, dict[str, float]] = {}
    option_letters = parse_option_letters(trace)
    for condition in CONDITIONS:
        logits, ids = target_logits(
            model,
            processor,
            image,
            context_prefix,
            target,
            evidence[condition],
            args.device,
        )
        if target_ids is not None and not torch.equal(target_ids, ids):
            raise ValueError(f"Target token mismatch in condition {condition}")
        target_ids = ids
        logits_by_condition[condition] = logits
        option_probs[condition] = option_probabilities(
            model,
            processor,
            image,
            context_prefix,
            evidence[condition],
            option_letters,
            args.device,
        )
    assert target_ids is not None
    opd = {
        condition: distribution_metrics(
            logits_by_condition["none"], logits_by_condition[condition], target_ids, args.top_k
        )
        for condition in CONDITIONS
    }
    opd_vs_blank = {
        condition: distribution_metrics(
            logits_by_condition["blank"], logits_by_condition[condition], target_ids, args.top_k
        )
        for condition in CONDITIONS
    }
    opd_vs_random = {
        condition: distribution_metrics(
            logits_by_condition["random"], logits_by_condition[condition], target_ids, args.top_k
        )
        for condition in CONDITIONS
    }
    gold = trace["ground_truth_label"]
    predicted = trace["predicted_label"]
    for condition in CONDITIONS:
        probabilities = option_probs[condition]
        opd[condition]["option_probabilities"] = probabilities
        opd[condition]["gold_probability"] = probabilities[gold]
        opd[condition]["predicted_probability"] = probabilities[predicted]
        opd[condition]["forced_choice_prediction"] = max(probabilities, key=probabilities.get)
        opd[condition]["forced_choice_correct"] = max(probabilities, key=probabilities.get) == gold

    result = {
        "index": index,
        "category": trace["category"],
        "rollout_correct": bool(trace["correct"]),
        "gold": gold,
        "rollout_prediction": predicted,
        "question": trace["question"],
        "target_objects": annotation["target_object"],
        "target_boxes": [list(box) for box in boxes],
        "image_size": list(image.size),
        "grid": list(grid),
        "selected_sentence_index": selected_id,
        "selected_sentence": target,
        "selected_sentence_image_mass": float(layer_masses[selected_id].mean()),
        "attention_input_mode": attention_input_mode,
        "stable_denominator_floor": denominator_floor,
        "localization": localization,
        "random_localization_baseline": random_baselines,
        "random_evidence_window": list(random_window),
        "oracle_window": list(oracle),
        "opd": opd,
        "opd_vs_blank": opd_vs_blank,
        "opd_vs_random": opd_vs_random,
        "probe_version": 2,
        "runtime_seconds": time.time() - started,
        "artifact_dir": str(sample_dir.resolve()),
    }
    return result


def mean(records: list[dict[str, Any]], getter: Any) -> float:
    values = [float(getter(record)) for record in records]
    return float(np.mean(values)) if values else float("nan")


def bootstrap_ci(values: list[float], seed: int, draws: int = 4000) -> list[float]:
    if not values:
        return [float("nan"), float("nan")]
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(array, size=(draws, len(array)), replace=True), axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def summarize_records(records: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "sample_count": len(records),
        "strata": {},
        "localization": {},
        "opd": {},
    }
    for category in sorted({record["category"] for record in records}):
        for correct in (False, True):
            subset = [r for r in records if r["category"] == category and r["rollout_correct"] == correct]
            summary["strata"][f"{category}:{'correct' if correct else 'wrong'}"] = len(subset)
    for method in METHODS:
        all_hit = [float(r["localization"][method]["all_target_centers_inside"]) for r in records]
        random_hit = [float(r["random_localization_baseline"][method]["all_target_centers_inside"]) for r in records]
        coverage = [float(r["localization"][method]["mean_bbox_coverage"]) for r in records]
        random_coverage = [float(r["random_localization_baseline"][method]["mean_bbox_coverage"]) for r in records]
        summary["localization"][method] = {
            "all_target_centers_hit_rate": float(np.mean(all_hit)),
            "all_target_centers_hit_rate_ci95": bootstrap_ci(all_hit, args.seed + 1),
            "random_same_size_hit_rate": float(np.mean(random_hit)),
            "hit_rate_lift_over_random": float(np.mean(np.asarray(all_hit) - np.asarray(random_hit))),
            "any_target_center_hit_rate": mean(records, lambda r: r["localization"][method]["any_target_center_inside"]),
            "mean_bbox_coverage": float(np.mean(coverage)),
            "random_same_size_bbox_coverage": float(np.mean(random_coverage)),
            "bbox_coverage_lift_over_random": float(np.mean(np.asarray(coverage) - np.asarray(random_coverage))),
            "peak_in_target_bbox_rate": mean(records, lambda r: r["localization"][method]["peak_in_target_bbox"]),
            "mean_window_center_distance": mean(records, lambda r: r["localization"][method]["window_center_distance"]),
            "mean_window_fraction": mean(records, lambda r: r["localization"][method]["window_fraction"]),
        }
    for condition in CONDITIONS:
        gold_deltas = [
            r["opd"][condition]["gold_probability"] - r["opd"]["none"]["gold_probability"]
            for r in records
        ]
        summary["opd"][condition] = {
            "teacher_to_student_kl": mean(records, lambda r: r["opd"][condition]["teacher_to_student_kl"]),
            "direct_kl_vs_blank": mean(records, lambda r: r["opd_vs_blank"][condition]["teacher_to_student_kl"]),
            "direct_jsd_vs_blank": mean(records, lambda r: r["opd_vs_blank"][condition]["jsd"]),
            "direct_topk_overlap_vs_blank": mean(records, lambda r: r["opd_vs_blank"][condition]["topk_overlap"]),
            "direct_teacher_mass_on_blank_topk": mean(records, lambda r: r["opd_vs_blank"][condition]["teacher_mass_on_student_topk"]),
            "direct_target_nll_delta_vs_blank": mean(records, lambda r: r["opd_vs_blank"][condition]["target_nll_delta"]),
            "direct_kl_vs_random": mean(records, lambda r: r["opd_vs_random"][condition]["teacher_to_student_kl"]),
            "jsd": mean(records, lambda r: r["opd"][condition]["jsd"]),
            "topk_overlap": mean(records, lambda r: r["opd"][condition]["topk_overlap"]),
            "teacher_mass_on_student_topk": mean(records, lambda r: r["opd"][condition]["teacher_mass_on_student_topk"]),
            "target_nll_delta": mean(records, lambda r: r["opd"][condition]["target_nll_delta"]),
            "gold_probability_delta": float(np.mean(gold_deltas)),
            "gold_probability_delta_ci95": bootstrap_ci(gold_deltas, args.seed + 2),
            "forced_choice_accuracy": mean(records, lambda r: r["opd"][condition]["forced_choice_correct"]),
            "wrong_rollout_gold_probability_delta": mean(
                [r for r in records if not r["rollout_correct"]],
                lambda r: r["opd"][condition]["gold_probability"] - r["opd"]["none"]["gold_probability"],
            ),
            "correct_rollout_gold_probability_delta": mean(
                [r for r in records if r["rollout_correct"]],
                lambda r: r["opd"][condition]["gold_probability"] - r["opd"]["none"]["gold_probability"],
            ),
        }
    summary["comparisons"] = {
        "stable_minus_blank_kl": summary["opd"]["relative_stable"]["teacher_to_student_kl"] - summary["opd"]["blank"]["teacher_to_student_kl"],
        "stable_minus_random_kl": summary["opd"]["relative_stable"]["teacher_to_student_kl"] - summary["opd"]["random"]["teacher_to_student_kl"],
        "stable_minus_blank_gold_delta": summary["opd"]["relative_stable"]["gold_probability_delta"] - summary["opd"]["blank"]["gold_probability_delta"],
        "stable_minus_random_gold_delta": summary["opd"]["relative_stable"]["gold_probability_delta"] - summary["opd"]["random"]["gold_probability_delta"],
        "oracle_minus_stable_gold_delta": summary["opd"]["oracle"]["gold_probability_delta"] - summary["opd"]["relative_stable"]["gold_probability_delta"],
    }
    conditional_groups = {
        "rollout_wrong": [r for r in records if not r["rollout_correct"]],
        "rollout_correct": [r for r in records if r["rollout_correct"]],
        "stable_hit_and_rollout_wrong": [
            r
            for r in records
            if not r["rollout_correct"]
            and r["localization"]["relative_stable"]["all_target_centers_inside"]
        ],
        "stable_miss_and_rollout_wrong": [
            r
            for r in records
            if not r["rollout_correct"]
            and not r["localization"]["relative_stable"]["all_target_centers_inside"]
        ],
    }
    summary["conditional"] = {}
    for group_name, subset in conditional_groups.items():
        stable_gold = [
            r["opd"]["relative_stable"]["gold_probability"]
            - r["opd"]["blank"]["gold_probability"]
            for r in subset
        ]
        oracle_gold = [
            r["opd"]["oracle"]["gold_probability"]
            - r["opd"]["blank"]["gold_probability"]
            for r in subset
        ]
        summary["conditional"][group_name] = {
            "sample_count": len(subset),
            "stable_gold_probability_delta_vs_blank": float(np.mean(stable_gold)),
            "stable_gold_probability_delta_vs_blank_ci95": bootstrap_ci(
                stable_gold, args.seed + 11
            ),
            "oracle_gold_probability_delta_vs_blank": float(np.mean(oracle_gold)),
            "oracle_gold_probability_delta_vs_blank_ci95": bootstrap_ci(
                oracle_gold, args.seed + 12
            ),
            "stable_direct_kl_vs_blank": mean(
                subset,
                lambda r: r["opd_vs_blank"]["relative_stable"]["teacher_to_student_kl"],
            ),
            "stable_target_nll_delta_vs_blank": mean(
                subset,
                lambda r: r["opd_vs_blank"]["relative_stable"]["target_nll_delta"],
            ),
        }
    summary["forced_choice_flips_stable_vs_blank"] = {
        "wrong_to_correct": sum(
            r["opd"]["blank"]["forced_choice_prediction"] != r["gold"]
            and r["opd"]["relative_stable"]["forced_choice_prediction"] == r["gold"]
            for r in records
        ),
        "correct_to_wrong": sum(
            r["opd"]["blank"]["forced_choice_prediction"] == r["gold"]
            and r["opd"]["relative_stable"]["forced_choice_prediction"] != r["gold"]
            for r in records
        ),
    }
    return summary


def read_all_shards(output_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(output_dir.glob("records.shard-*.jsonl")):
        records.extend(json.loads(line) for line in path.open(encoding="utf-8") if line.strip())
    unique = {int(record["index"]): record for record in records}
    return [unique[index] for index in sorted(unique)]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        records = read_all_shards(args.output_dir)
        summary = summarize_records(records, args)
        (args.output_dir / "records.jsonl").write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
            encoding="utf-8",
        )
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    traces = load_trace_records(args.traces)
    selected = select_indices(traces, args.indices, args.pilot_per_stratum, args.seed)
    shard_indices = [index for position, index in enumerate(selected) if position % args.num_shards == args.shard_id]
    annotations = load_annotations(args.annotations)
    table = pq.read_table(args.data)
    rows = table.to_pylist()

    processor = AutoProcessor.from_pretrained(args.model)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(args.device)
    model.eval()

    output_path = args.output_dir / f"records.shard-{args.shard_id:02d}.jsonl"
    completed: dict[int, dict[str, Any]] = {}
    if output_path.exists():
        completed = {
            int(record["index"]): record
            for record in (json.loads(line) for line in output_path.open(encoding="utf-8") if line.strip())
        }
    with output_path.open("a", encoding="utf-8") as handle:
        for position, index in enumerate(shard_indices, start=1):
            if index in completed and completed[index].get("probe_version", 1) >= 2:
                print(f"[{position}/{len(shard_indices)}] index={index} already complete", flush=True)
                continue
            trace = traces[index]
            key = normalized_question(trace["question"])
            if key not in annotations:
                raise KeyError(f"No official annotation for index {index}: {key}")
            print(f"[{position}/{len(shard_indices)}] index={index} starting", flush=True)
            result = analyze_one(model, processor, rows[index], trace, annotations[key], args)
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"[{position}/{len(shard_indices)}] index={index} done "
                f"({result['runtime_seconds']:.1f}s)",
                flush=True,
            )


if __name__ == "__main__":
    main()
