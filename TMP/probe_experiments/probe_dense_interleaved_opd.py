#!/usr/bin/env python3
"""Dense no-training probe with hindsight evidence E_i inserted before R_i.

Student: [V,Q] R1 R2 ... RT
Teacher: [V,Q] E1 R1 E2 R2 ... ET RT

Each E_i is selected offline from the already-completed Student rollout using
R_i attention.  The frozen same-model Teacher scores the identical R_i tokens
with privileged interleaved visual context; no parameter update is performed.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from analyze_vstar_attention import (
    capture_full_attention,
    find_subsequence,
    load_trace_records,
    prepare_descriptive_inputs,
    prepare_teacher_forced_inputs,
    relative_attention,
)
from probe_interleaved_opd import (
    DEFAULT_ANNOTATIONS,
    DEFAULT_DATA,
    DEFAULT_MODEL,
    DEFAULT_TRACES,
    IMAGE_MARKER,
    annotation_boxes,
    best_sliding_window,
    bootstrap_ci,
    crop_score_map,
    deterministic_rng,
    distribution_metrics,
    grid_window_to_pixels,
    load_annotations,
    localization_metrics,
    mean,
    oracle_window,
    parse_option_letters,
    prepare_retokenized_teacher_inputs,
    probability_map,
    save_localization_panel,
    select_indices,
    stable_relative_attention,
)


DEFAULT_OUTPUT = Path(
    "/root/siton-tmp/yzs/mmcot_opsd/outputs/dense-interleaved-opd-probe"
)
METHODS = ("absolute", "relative_strict", "relative_stable")
CONDITIONS = (
    "none",
    "dense_blank",
    "dense_random",
    "dense_absolute",
    "dense_strict",
    "dense_stable",
    "dense_oracle",
)
STEP_OPTION_CONDITIONS = (
    "none",
    "dense_blank",
    "dense_random",
    "dense_stable",
    "dense_oracle",
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
    parser.add_argument("--summarize-only", action="store_true")
    return parser.parse_args()


def find_subsequence_from(haystack: list[int], needle: list[int], start: int) -> int:
    for index in range(start, len(haystack) - len(needle) + 1):
        if haystack[index : index + len(needle)] == needle:
            return index
    raise ValueError("Aligned sentence-token suffix was not found")


def build_dense_text(
    prompt: str,
    completion: str,
    sentence_specs: list[dict[str, Any]],
) -> str:
    parts = [prompt]
    cursor = 0
    for spec in sentence_specs:
        parts.append(completion[cursor : spec["char_start"]])
        parts.append(IMAGE_MARKER)
        parts.append("\n")
        parts.append(spec["text"])
        cursor = spec["char_end"]
    parts.append(completion[cursor:])
    return "".join(parts)


def build_dense_prefix(
    prompt: str,
    completion: str,
    sentence_specs: list[dict[str, Any]],
    decision_position: int,
) -> str:
    parts = [prompt]
    cursor = 0
    for position, spec in enumerate(sentence_specs):
        if position > decision_position:
            break
        parts.append(completion[cursor : spec["char_start"]])
        parts.append(IMAGE_MARKER)
        parts.append("\n")
        if position == decision_position:
            break
        parts.append(spec["text"])
        cursor = spec["char_end"]
    return "".join(parts)


def aligned_target_layout(
    processor: Any,
    input_ids: list[int],
    sentence_specs: list[dict[str, Any]],
) -> tuple[list[int], list[int], list[dict[str, int]]]:
    target_ids: list[int] = []
    query_positions: list[int] = []
    layout: list[dict[str, int]] = []
    search_cursor = 0
    output_cursor = 0
    for spec in sentence_specs:
        full_ids = processor.tokenizer(spec["text"], add_special_tokens=False)["input_ids"]
        # Drop the sentence-initial boundary token.  Its ID can differ between
        # the original whitespace boundary and an inserted-image newline.
        aligned_ids = full_ids[1:]
        if not aligned_ids:
            continue
        start = find_subsequence_from(input_ids, aligned_ids, search_cursor)
        query_positions.extend(range(start - 1, start + len(aligned_ids) - 1))
        target_ids.extend(aligned_ids)
        layout.append(
            {
                "sentence_index": int(spec["sentence_index"]),
                "start": output_cursor,
                "end": output_cursor + len(aligned_ids),
                "tokens": len(aligned_ids),
            }
        )
        output_cursor += len(aligned_ids)
        search_cursor = start + len(aligned_ids)
    return target_ids, query_positions, layout


def dense_target_logits(
    model: Qwen3_5ForConditionalGeneration,
    processor: Any,
    original: Image.Image,
    text: str,
    evidence: list[Image.Image] | None,
    sentence_specs: list[dict[str, Any]],
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, int]], list[str]]:
    images = [original] if evidence is None else [original, *evidence]
    batch = processor(text=[text], images=images, return_tensors="pt")
    target_ids, query_positions, layout = aligned_target_layout(
        processor, batch["input_ids"][0].tolist(), sentence_specs
    )
    position_tensor = torch.tensor(query_positions, dtype=torch.long, device=device)
    batch = {key: value.to(device) for key, value in batch.items()}
    model.config.text_config._attn_implementation = "flash_attention_2"
    with torch.inference_mode():
        logits = model(
            **batch,
            use_cache=False,
            logits_to_keep=position_tensor,
        ).logits[0].float().cpu()
    decoded = [processor.tokenizer.decode([token_id]) for token_id in target_ids]
    return logits, torch.tensor(target_ids, dtype=torch.long), layout, decoded


def option_probabilities_at_decision(
    model: Qwen3_5ForConditionalGeneration,
    processor: Any,
    original: Image.Image,
    text_prefix: str,
    evidence: list[Image.Image] | None,
    options: list[str],
    device: str,
) -> dict[str, float]:
    suffix = "The correct answer is ("
    images = [original] if evidence is None else [original, *evidence]
    batch = processor(text=[text_prefix + suffix], images=images, return_tensors="pt")
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
    probabilities = torch.softmax(logits[candidate_ids], dim=0).cpu().tolist()
    return {option: float(value) for option, value in zip(options, probabilities)}


def decision_position(sentence_specs: list[dict[str, Any]]) -> int:
    pattern = re.compile(r"\b(?:therefore|thus|hence|correct answer|answer is)\b", re.I)
    for position, spec in enumerate(sentence_specs):
        if pattern.search(spec["text"]):
            return position
    return max(len(sentence_specs) - 1, 0)


def resize_evidence(crop: Image.Image, size: tuple[int, int]) -> Image.Image:
    if crop.size == size:
        return crop
    return crop.resize(size, Image.Resampling.LANCZOS)


def content_mask(decoded_tokens: list[str]) -> np.ndarray:
    return np.asarray([bool(re.search(r"[A-Za-z0-9]", token)) for token in decoded_tokens])


def add_signal_density(
    metrics: dict[str, Any],
    decoded_tokens: list[str],
) -> dict[str, Any]:
    token_kl = np.asarray(metrics["token_teacher_to_student_kl"], dtype=np.float64)
    content = content_mask(decoded_tokens)
    metrics["token_fraction_kl_gt_0_01"] = float(np.mean(token_kl > 0.01))
    metrics["token_fraction_kl_gt_0_05"] = float(np.mean(token_kl > 0.05))
    metrics["token_fraction_kl_gt_0_10"] = float(np.mean(token_kl > 0.10))
    metrics["content_token_fraction"] = float(np.mean(content))
    metrics["content_token_kl"] = float(np.mean(token_kl[content])) if content.any() else float("nan")
    metrics["noncontent_token_kl"] = (
        float(np.mean(token_kl[~content])) if (~content).any() else float("nan")
    )
    return metrics


def per_sentence_metrics(
    blank_logits: torch.Tensor,
    condition_logits: torch.Tensor,
    target_ids: torch.Tensor,
    layout: list[dict[str, int]],
    top_k: int,
) -> list[dict[str, Any]]:
    values = []
    for item in layout:
        start, end = item["start"], item["end"]
        metrics = distribution_metrics(
            blank_logits[start:end],
            condition_logits[start:end],
            target_ids[start:end],
            top_k,
        )
        values.append(
            {
                "sentence_index": item["sentence_index"],
                "tokens": item["tokens"],
                "direct_kl_vs_blank": metrics["teacher_to_student_kl"],
                "topk_overlap_vs_blank": metrics["topk_overlap"],
                "target_nll_delta_vs_blank": metrics["target_nll_delta"],
            }
        )
    return values


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
        attention_mode = "recorded_token_ids"
    except ValueError as error:
        if "Completion token round-trip mismatch" not in str(error):
            raise
        inputs, sentence_specs, image_positions, grid = prepare_retokenized_teacher_inputs(
            processor, image, trace, args.device
        )
        attention_mode = "retokenized_completion_fallback"
    if not sentence_specs:
        raise ValueError(f"No reasoning sentences for index {index}")

    question_ids = processor.tokenizer(trace["question"], add_special_tokens=False)["input_ids"]
    question_start = find_subsequence(inputs["input_ids"][0].tolist(), question_ids)
    question_spec = {
        "sentence_index": -1,
        "text": trace["question"],
        "query_positions": [question_start + offset - 1 for offset in range(len(question_ids))],
    }
    layer_maps, layer_masses = capture_full_attention(
        model, inputs, [question_spec, *sentence_specs], image_positions
    )
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
    families: dict[str, dict[int, np.ndarray]] = {
        "absolute": {key: probability_map(value) for key, value in raw_maps.items()},
        "relative_strict": {
            key: relative_attention(value, descriptive)[1] for key, value in raw_maps.items()
        },
        "relative_stable": {
            key: stable_relative_attention(value, descriptive, grid)[1]
            for key, value in raw_maps.items()
        },
    }

    windows_by_sentence: list[dict[str, tuple[int, int, int, int]]] = []
    heatmaps_by_sentence: list[dict[str, np.ndarray]] = []
    localization_by_sentence: list[dict[str, Any]] = []
    prefix_raw: np.ndarray | None = None
    prefix_weight = 0.0
    for spec in sentence_specs:
        sentence_id = int(spec["sentence_index"])
        heatmaps: dict[str, np.ndarray] = {}
        windows: dict[str, tuple[int, int, int, int]] = {}
        localization: dict[str, Any] = {}
        for method, maps in families.items():
            if prefix_raw is None:
                prefix_map = None
            elif method == "absolute":
                prefix_map = probability_map(prefix_raw)
            elif method == "relative_strict":
                prefix_map = relative_attention(prefix_raw, descriptive)[1]
            else:
                prefix_map = stable_relative_attention(prefix_raw, descriptive, grid)[1]
            heat = crop_score_map(maps[-1], prefix_map, maps[sentence_id])
            window = grid_window_to_pixels(best_sliding_window(heat, grid), grid, image.size)
            peak_index = int(np.argmax(heat))
            peak_row, peak_col = divmod(peak_index, grid[1])
            peak = (
                (peak_col + 0.5) * image.width / grid[1],
                (peak_row + 0.5) * image.height / grid[0],
            )
            heatmaps[method] = heat
            windows[method] = window
            localization[method] = localization_metrics(window, peak, boxes, image.size)
        heatmaps_by_sentence.append(heatmaps)
        windows_by_sentence.append(windows)
        localization_by_sentence.append(localization)
        sentence_weight = float(len(spec["query_positions"]))
        if prefix_raw is None:
            prefix_raw = raw_maps[sentence_id].copy()
            prefix_weight = sentence_weight
        else:
            prefix_raw = (
                prefix_raw * prefix_weight + raw_maps[sentence_id] * sentence_weight
            ) / (prefix_weight + sentence_weight)
            prefix_weight += sentence_weight

    oracle = oracle_window(boxes, image.size)
    mean_color = tuple(int(value) for value in np.asarray(image).reshape(-1, 3).mean(axis=0))
    evidence: dict[str, list[Image.Image]] = {
        condition: [] for condition in CONDITIONS if condition != "none"
    }
    for position, spec in enumerate(sentence_specs):
        stable_window = windows_by_sentence[position]["relative_stable"]
        canonical_size = (
            stable_window[2] - stable_window[0],
            stable_window[3] - stable_window[1],
        )
        rng = deterministic_rng(args.seed, index, f"dense-random-{position}")
        left = int(rng.integers(0, max(image.width - canonical_size[0], 0) + 1))
        top = int(rng.integers(0, max(image.height - canonical_size[1], 0) + 1))
        random_crop = image.crop((left, top, left + canonical_size[0], top + canonical_size[1]))
        evidence["dense_blank"].append(Image.new("RGB", canonical_size, mean_color))
        evidence["dense_random"].append(random_crop)
        evidence["dense_absolute"].append(
            resize_evidence(image.crop(windows_by_sentence[position]["absolute"]), canonical_size)
        )
        evidence["dense_strict"].append(
            resize_evidence(image.crop(windows_by_sentence[position]["relative_strict"]), canonical_size)
        )
        evidence["dense_stable"].append(image.crop(stable_window))
        evidence["dense_oracle"].append(resize_evidence(image.crop(oracle), canonical_size))

    decision = decision_position(sentence_specs)
    save_localization_panel(
        image,
        grid,
        heatmaps_by_sentence[decision],
        windows_by_sentence[decision],
        boxes,
        sample_dir / "decision-localization.jpg",
    )
    for condition in ("dense_stable", "dense_oracle"):
        evidence[condition][decision].save(
            sample_dir / f"decision-{condition}.jpg", quality=93
        )

    student_text = trace["prompt"] + trace["completion"]
    dense_text = build_dense_text(trace["prompt"], trace["completion"], sentence_specs)
    logits_by_condition: dict[str, torch.Tensor] = {}
    target_ids: torch.Tensor | None = None
    layout: list[dict[str, int]] | None = None
    decoded_tokens: list[str] | None = None
    for condition in CONDITIONS:
        condition_text = student_text if condition == "none" else dense_text
        condition_evidence = None if condition == "none" else evidence[condition]
        logits, ids, condition_layout, decoded = dense_target_logits(
            model,
            processor,
            image,
            condition_text,
            condition_evidence,
            sentence_specs,
            args.device,
        )
        if target_ids is not None and not torch.equal(target_ids, ids):
            raise ValueError(f"Dense target mismatch for index {index}, condition {condition}")
        target_ids = ids
        layout = condition_layout
        decoded_tokens = decoded
        logits_by_condition[condition] = logits
    assert target_ids is not None and layout is not None and decoded_tokens is not None

    opd = {}
    opd_vs_blank = {}
    for condition in CONDITIONS:
        opd[condition] = add_signal_density(
            distribution_metrics(
                logits_by_condition["none"],
                logits_by_condition[condition],
                target_ids,
                args.top_k,
            ),
            decoded_tokens,
        )
        opd_vs_blank[condition] = add_signal_density(
            distribution_metrics(
                logits_by_condition["dense_blank"],
                logits_by_condition[condition],
                target_ids,
                args.top_k,
            ),
            decoded_tokens,
        )
        opd_vs_blank[condition]["per_sentence"] = per_sentence_metrics(
            logits_by_condition["dense_blank"],
            logits_by_condition[condition],
            target_ids,
            layout,
            args.top_k,
        )

    options = parse_option_letters(trace)
    gold = trace["ground_truth_label"]
    step_option_probes: dict[str, list[dict[str, Any]]] = {
        condition: [] for condition in STEP_OPTION_CONDITIONS
    }
    for step_position, spec in enumerate(sentence_specs):
        student_prefix = trace["prompt"] + trace["completion"][: spec["char_start"]]
        dense_prefix = build_dense_prefix(
            trace["prompt"], trace["completion"], sentence_specs, step_position
        )
        for condition in STEP_OPTION_CONDITIONS:
            if condition == "none":
                prefix = student_prefix
                condition_evidence = None
            else:
                prefix = dense_prefix
                condition_evidence = evidence[condition][: step_position + 1]
            probabilities = option_probabilities_at_decision(
                model,
                processor,
                image,
                prefix,
                condition_evidence,
                options,
                args.device,
            )
            prediction = max(probabilities, key=probabilities.get)
            step_option_probes[condition].append(
                {
                    "sentence_index": int(spec["sentence_index"]),
                    "gold_probability": probabilities[gold],
                    "prediction": prediction,
                    "correct": prediction == gold,
                }
            )

    for condition in CONDITIONS:
        if condition in STEP_OPTION_CONDITIONS:
            decision_probe = step_option_probes[condition][decision]
            probabilities = None
        else:
            dense_decision_prefix = build_dense_prefix(
                trace["prompt"], trace["completion"], sentence_specs, decision
            )
            probabilities = option_probabilities_at_decision(
                model,
                processor,
                image,
                dense_decision_prefix,
                evidence[condition][: decision + 1],
                options,
                args.device,
            )
            prediction = max(probabilities, key=probabilities.get)
            decision_probe = {
                "gold_probability": probabilities[gold],
                "prediction": prediction,
                "correct": prediction == gold,
            }
        if probabilities is not None:
            opd[condition]["decision_option_probabilities"] = probabilities
        opd[condition]["decision_gold_probability"] = decision_probe["gold_probability"]
        opd[condition]["decision_prediction"] = decision_probe["prediction"]
        opd[condition]["decision_correct"] = decision_probe["correct"]

    stable_hits = [
        bool(value["relative_stable"]["all_target_centers_inside"])
        for value in localization_by_sentence
    ]
    result = {
        "probe_version": 2,
        "index": index,
        "category": trace["category"],
        "rollout_correct": bool(trace["correct"]),
        "gold": trace["ground_truth_label"],
        "rollout_prediction": trace["predicted_label"],
        "question": trace["question"],
        "sentence_count": len(sentence_specs),
        "aligned_target_tokens": int(target_ids.numel()),
        "attention_input_mode": attention_mode,
        "decision_position": decision,
        "decision_sentence_index": int(sentence_specs[decision]["sentence_index"]),
        "decision_sentence": sentence_specs[decision]["text"],
        "stable_window_hit_fraction": float(np.mean(stable_hits)),
        "stable_decision_window_hit": stable_hits[decision],
        "localization_by_sentence": localization_by_sentence,
        "step_option_probes": step_option_probes,
        "opd": opd,
        "opd_vs_blank": opd_vs_blank,
        "runtime_seconds": time.time() - started,
        "artifact_dir": str(sample_dir.resolve()),
    }
    return result


def read_records(output_dir: Path) -> list[dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for path in sorted(output_dir.glob("records.shard-*.jsonl")):
        for line in path.open(encoding="utf-8"):
            if line.strip():
                record = json.loads(line)
                records[int(record["index"])] = record
    return [records[index] for index in sorted(records)]


def summarize(records: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "sample_count": len(records),
        "mean_sentences_per_sample": mean(records, lambda r: r["sentence_count"]),
        "mean_aligned_tokens_per_sample": mean(records, lambda r: r["aligned_target_tokens"]),
        "conditions": {},
    }
    for condition in CONDITIONS:
        gold_vs_blank = [
            r["opd"][condition]["decision_gold_probability"]
            - r["opd"]["dense_blank"]["decision_gold_probability"]
            for r in records
        ]
        wrong = [r for r in records if not r["rollout_correct"]]
        correct = [r for r in records if r["rollout_correct"]]
        summary["conditions"][condition] = {
            "direct_kl_vs_blank": mean(
                records, lambda r: r["opd_vs_blank"][condition]["teacher_to_student_kl"]
            ),
            "direct_jsd_vs_blank": mean(
                records, lambda r: r["opd_vs_blank"][condition]["jsd"]
            ),
            "token_fraction_direct_kl_gt_0_05": mean(
                records, lambda r: r["opd_vs_blank"][condition]["token_fraction_kl_gt_0_05"]
            ),
            "content_token_direct_kl_vs_blank": mean(
                records, lambda r: r["opd_vs_blank"][condition]["content_token_kl"]
            ),
            "topk_overlap_vs_blank": mean(
                records, lambda r: r["opd_vs_blank"][condition]["topk_overlap"]
            ),
            "teacher_mass_on_student_topk": mean(
                records, lambda r: r["opd"][condition]["teacher_mass_on_student_topk"]
            ),
            "decision_gold_delta_vs_blank": float(np.mean(gold_vs_blank)),
            "decision_gold_delta_vs_blank_ci95": bootstrap_ci(gold_vs_blank, args.seed + 20),
            "decision_accuracy": mean(records, lambda r: r["opd"][condition]["decision_correct"]),
            "wrong_rollout_gold_delta_vs_blank": mean(
                wrong,
                lambda r: r["opd"][condition]["decision_gold_probability"]
                - r["opd"]["dense_blank"]["decision_gold_probability"],
            ),
            "correct_rollout_gold_delta_vs_blank": mean(
                correct,
                lambda r: r["opd"][condition]["decision_gold_probability"]
                - r["opd"]["dense_blank"]["decision_gold_probability"],
            ),
        }
    wrong = [r for r in records if not r["rollout_correct"]]
    summary["conditional"] = {}
    for name, subset in {
        "wrong_decision_hit": [r for r in wrong if r["stable_decision_window_hit"]],
        "wrong_decision_miss": [r for r in wrong if not r["stable_decision_window_hit"]],
        "wrong_majority_windows_hit": [r for r in wrong if r["stable_window_hit_fraction"] >= 0.5],
        "wrong_minority_windows_hit": [r for r in wrong if r["stable_window_hit_fraction"] < 0.5],
    }.items():
        values = [
            r["opd"]["dense_stable"]["decision_gold_probability"]
            - r["opd"]["dense_blank"]["decision_gold_probability"]
            for r in subset
        ]
        summary["conditional"][name] = {
            "sample_count": len(subset),
            "stable_gold_delta_vs_blank": float(np.mean(values)) if values else float("nan"),
            "stable_gold_delta_vs_blank_ci95": bootstrap_ci(values, args.seed + 21),
        }
    summary["stable_localization"] = {
        "mean_sentence_window_hit_fraction": mean(records, lambda r: r["stable_window_hit_fraction"]),
        "decision_window_hit_rate": mean(records, lambda r: r["stable_decision_window_hit"]),
    }
    summary["forced_choice_flips_stable_vs_blank"] = {
        "wrong_to_correct": sum(
            not r["opd"]["dense_blank"]["decision_correct"]
            and r["opd"]["dense_stable"]["decision_correct"]
            for r in records
        ),
        "correct_to_wrong": sum(
            r["opd"]["dense_blank"]["decision_correct"]
            and not r["opd"]["dense_stable"]["decision_correct"]
            for r in records
        ),
    }
    summary["stepwise_option_signal"] = {}
    for condition in STEP_OPTION_CONDITIONS:
        if condition == "dense_blank":
            deltas = [[0.0] * r["sentence_count"] for r in records]
        else:
            deltas = [
                [
                    probe["gold_probability"] - blank["gold_probability"]
                    for probe, blank in zip(
                        r["step_option_probes"][condition],
                        r["step_option_probes"]["dense_blank"],
                    )
                ]
                for r in records
            ]
        per_sample_mean = [float(np.mean(values)) for values in deltas]
        first_step = [values[0] for values in deltas]
        positive_fraction = [float(np.mean(np.asarray(values) > 0)) for values in deltas]
        wrong_positions = [position for position, r in enumerate(records) if not r["rollout_correct"]]
        correct_positions = [position for position, r in enumerate(records) if r["rollout_correct"]]
        wrong_mean = [per_sample_mean[position] for position in wrong_positions]
        wrong_first = [first_step[position] for position in wrong_positions]
        correct_mean = [per_sample_mean[position] for position in correct_positions]
        summary["stepwise_option_signal"][condition] = {
            "mean_step_gold_delta_vs_blank": float(np.mean(per_sample_mean)),
            "mean_step_gold_delta_vs_blank_ci95": bootstrap_ci(
                per_sample_mean, args.seed + 30
            ),
            "first_step_gold_delta_vs_blank": float(np.mean(first_step)),
            "positive_step_fraction": float(np.mean(positive_fraction)),
            "wrong_rollout_mean_step_gold_delta_vs_blank": float(np.mean(wrong_mean)),
            "wrong_rollout_mean_step_gold_delta_vs_blank_ci95": bootstrap_ci(
                wrong_mean, args.seed + 31
            ),
            "wrong_rollout_first_step_gold_delta_vs_blank": float(np.mean(wrong_first)),
            "wrong_rollout_first_step_gold_delta_vs_blank_ci95": bootstrap_ci(
                wrong_first, args.seed + 32
            ),
            "correct_rollout_mean_step_gold_delta_vs_blank": float(np.mean(correct_mean)),
        }
    return summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        records = [
            record
            for record in read_records(args.output_dir)
            if record.get("probe_version", 1) >= 2
        ]
        result = summarize(records, args)
        (args.output_dir / "records.jsonl").write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
            encoding="utf-8",
        )
        (args.output_dir / "summary.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    traces = load_trace_records(args.traces)
    selected = select_indices(traces, args.indices, args.pilot_per_stratum, args.seed)
    shard_indices = [
        index
        for position, index in enumerate(selected)
        if position % args.num_shards == args.shard_id
    ]
    annotations = load_annotations(args.annotations)
    rows = pq.read_table(args.data).to_pylist()
    processor = AutoProcessor.from_pretrained(args.model)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(args.device)
    model.eval()

    output_path = args.output_dir / f"records.shard-{args.shard_id:02d}.jsonl"
    completed = {}
    if output_path.exists():
        completed = {
            int(record["index"]): record
            for record in (
                json.loads(line) for line in output_path.open(encoding="utf-8") if line.strip()
            )
        }
    with output_path.open("a", encoding="utf-8") as handle:
        for position, index in enumerate(shard_indices, start=1):
            if index in completed and completed[index].get("probe_version", 1) >= 2:
                print(f"[{position}/{len(shard_indices)}] index={index} already complete", flush=True)
                continue
            trace = traces[index]
            key = re.sub(r"\s+", " ", trace["question"].splitlines()[0].strip()).rstrip(".").lower()
            print(f"[{position}/{len(shard_indices)}] index={index} starting", flush=True)
            result = analyze_one(model, processor, rows[index], trace, annotations[key], args)
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"[{position}/{len(shard_indices)}] index={index} done "
                f"({result['sentence_count']} sentences, {result['runtime_seconds']:.1f}s)",
                flush=True,
            )


if __name__ == "__main__":
    main()
