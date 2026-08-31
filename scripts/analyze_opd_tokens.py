#!/usr/bin/env python3
"""Decode sampled-token OPD dumps and summarize credit assignment."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoTokenizer


def quantiles(values: torch.Tensor) -> tuple[float, float, float]:
    result = torch.quantile(values.float(), torch.tensor([0.1, 0.5, 0.9]))
    return tuple(float(value) for value in result)


def marker_position(tokenizer, token_ids: list[int], marker: str = "FINAL:") -> int | None:
    for index in range(len(token_ids)):
        if marker in tokenizer.decode(token_ids[: index + 1], skip_special_tokens=True):
            return index
    return None


def fmt_triplet(values: tuple[float, float, float]) -> str:
    return " / ".join(f"{value:.4f}" for value in values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-dir", type=Path, required=True)
    parser.add_argument("--rollouts-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--step", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-contexts", type=int, default=30)
    args = parser.parse_args()

    files = sorted(args.dump_dir.glob("*.rank*.pt"))
    if not files:
        raise FileNotFoundError(f"No OPD token dumps in {args.dump_dir}")
    available_steps = sorted({int(path.name.split(".", 1)[0]) for path in files})
    step = args.step if args.step is not None else available_steps[-1]
    step_files = sorted(args.dump_dir.glob(f"{step}.rank*.pt"))
    if not step_files:
        raise FileNotFoundError(f"No rank dumps for step {step}")

    keys = (
        "student_log_probs",
        "teacher_log_probs",
        "token_ids",
        "response_positions",
        "sample_ids",
        "outcomes",
        "advantages",
        "teacher_gaps",
        "gates",
        "opd_gradient_weights",
        "rollout_log_probs",
        "ref_log_probs",
    )
    loaded = [torch.load(path, map_location="cpu", weights_only=False) for path in step_files]
    tensors = {
        key: torch.cat([item[key] for item in loaded if key in item], dim=0)
        for key in keys
        if any(key in item for item in loaded)
    }
    required = {"student_log_probs", "teacher_log_probs", "token_ids", "response_positions", "sample_ids"}
    missing = required - tensors.keys()
    if missing:
        raise KeyError(f"Token dump is missing {sorted(missing)}")

    gap = tensors.get("teacher_gaps", tensors["teacher_log_probs"] - tensors["student_log_probs"]).float()
    beta = float(loaded[0].get("gate_beta", 5.0))
    gate = tensors.get("gates", torch.sigmoid(beta * gap)).float()
    outcomes = tensors.get("outcomes")
    advantages = tensors.get("advantages")
    token_ids = tensors["token_ids"].long()
    positions = tensors["response_positions"].long()
    sample_ids = tensors["sample_ids"].long()

    rollout_file = args.rollouts_dir / f"{step}.jsonl"
    rollout_by_id: dict[int, dict] = {}
    if rollout_file.exists():
        for line in rollout_file.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if "rollout_sample_id" in row:
                rollout_by_id[int(row["rollout_sample_id"])] = row

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    sample_indices: dict[int, list[int]] = defaultdict(list)
    for index, sample_id in enumerate(sample_ids.tolist()):
        sample_indices[int(sample_id)].append(index)

    final_mask = torch.zeros_like(gate, dtype=torch.bool)
    reasoning_mask = torch.ones_like(gate, dtype=torch.bool)
    sample_text: dict[int, str] = {}
    for sample_id, indices in sample_indices.items():
        indices.sort(key=lambda idx: int(positions[idx]))
        ids = [int(token_ids[idx]) for idx in indices]
        sample_text[sample_id] = tokenizer.decode(ids, skip_special_tokens=True)
        marker = marker_position(tokenizer, ids)
        if marker is not None:
            for local_index, global_index in enumerate(indices):
                if local_index >= marker:
                    final_mask[global_index] = True
                    reasoning_mask[global_index] = False

    topk_count = max(1, math.ceil(0.1 * gate.numel()))
    top_indices = torch.topk(gate, k=min(args.top_contexts, gate.numel())).indices.tolist()
    gate_mass_top10 = float(torch.topk(gate, k=topk_count).values.sum() / gate.sum().clamp_min(1e-12))

    lines = [
        f"# OPD token credit report — step {step}",
        "",
        f"- Samples: {len(sample_indices)}",
        f"- Valid response tokens: {gate.numel()}",
        f"- Gate beta: {beta:g}",
        f"- Gate p10 / p50 / p90: {fmt_triplet(quantiles(gate))}",
        f"- Teacher gap p10 / p50 / p90: {fmt_triplet(quantiles(gap))}",
        f"- Positive-gap token fraction: {float((gap > 0).float().mean()):.2%}",
        f"- OPD gradient mass carried by top 10% gate tokens: {gate_mass_top10:.2%}",
    ]

    if outcomes is not None:
        correct = outcomes > 0.5
        incorrect = ~correct
        lines.extend(
            [
                f"- Mean gate, correct trajectories: {float(gate[correct].mean()):.4f}",
                f"- Mean gate, incorrect trajectories: {float(gate[incorrect].mean()):.4f}",
            ]
        )
    if final_mask.any():
        lines.extend(
            [
                f"- Mean gate, reasoning tokens: {float(gate[reasoning_mask].mean()):.4f}",
                f"- Mean gate, `FINAL:` suffix tokens: {float(gate[final_mask].mean()):.4f}",
                f"- Positive gap, `FINAL:` suffix tokens: {float((gap[final_mask] > 0).float().mean()):.2%}",
            ]
        )
        if outcomes is not None:
            for label, mask in (("correct", outcomes > 0.5), ("incorrect", outcomes <= 0.5)):
                selected = final_mask & mask
                if selected.any():
                    lines.append(f"- Mean final-token gate, {label}: {float(gate[selected].mean()):.4f}")

    if advantages is not None:
        advantages = advantages.float()
        alignment = float(
            torch.dot(gate, advantages)
            / (torch.linalg.vector_norm(gate) * torch.linalg.vector_norm(advantages)).clamp_min(1e-12)
        )
        positive_mass = float(gate[advantages > 0].sum() / gate.sum().clamp_min(1e-12))
        negative_mass = float(gate[advantages < 0].sum() / gate.sum().clamp_min(1e-12))
        opd_weight = tensors.get("opd_gradient_weights")
        if opd_weight is not None:
            opd_rms = torch.sqrt(torch.mean(opd_weight.float().square()))
            grpo_rms = torch.sqrt(torch.mean(advantages.square())).clamp_min(1e-12)
            lines.append(f"- OPD/GRPO log-prob gradient RMS proxy: {float(opd_rms / grpo_rms):.4%}")
        lines.extend(
            [
                f"- OPD–GRPO log-prob gradient cosine: {alignment:+.4f}",
                f"- OPD mass on positive-advantage tokens: {positive_mass:.2%}",
                f"- OPD mass on negative-advantage tokens: {negative_mass:.2%}",
            ]
        )

    lines.extend(["", "## Highest-gate token contexts", ""])
    for rank_index, global_index in enumerate(top_indices, 1):
        sample_id = int(sample_ids[global_index])
        ordered = sorted(sample_indices[sample_id], key=lambda idx: int(positions[idx]))
        local = ordered.index(global_index)
        left, right = max(0, local - 6), min(len(ordered), local + 7)
        context_ids = [int(token_ids[idx]) for idx in ordered[left:right]]
        context = tokenizer.decode(context_ids, skip_special_tokens=True).replace("\n", "\\n")
        token_text = tokenizer.decode([int(token_ids[global_index])], skip_special_tokens=True).replace("\n", "\\n")
        outcome = None if outcomes is None else bool(outcomes[global_index] > 0.5)
        rollout = rollout_by_id.get(sample_id, {})
        lines.append(
            f"{rank_index}. gate={float(gate[global_index]):.4f}, gap={float(gap[global_index]):+.4f}, "
            f"token=`{token_text}`, position={int(positions[global_index])}, correct={outcome}, "
            f"sample={sample_id}, predicted={rollout.get('predicted_label')} — `{context}`"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
