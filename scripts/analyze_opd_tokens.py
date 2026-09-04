#!/usr/bin/env python3
"""Decode signed sampled-token OPSD dumps and summarize credit assignment."""

from __future__ import annotations

import argparse
import json
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
        raise FileNotFoundError(f"No OPSD token dumps in {args.dump_dir}")
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
        "opsd_advantages",
        "opsd_gradient_weights",
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
    opsd_advantage = tensors.get("opsd_advantages")
    if opsd_advantage is None:
        opsd_advantage = gap.clone()
        advantage_clip = loaded[0].get("opsd_advantage_clip")
        if advantage_clip is not None:
            opsd_advantage = opsd_advantage.clamp(
                min=-float(advantage_clip), max=float(advantage_clip)
            )
    opsd_advantage = opsd_advantage.float()
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

    final_mask = torch.zeros_like(opsd_advantage, dtype=torch.bool)
    reasoning_mask = torch.ones_like(opsd_advantage, dtype=torch.bool)
    for sample_id, indices in sample_indices.items():
        indices.sort(key=lambda idx: int(positions[idx]))
        ids = [int(token_ids[idx]) for idx in indices]
        marker = marker_position(tokenizer, ids)
        if marker is not None:
            for local_index, global_index in enumerate(indices):
                if local_index >= marker:
                    final_mask[global_index] = True
                    reasoning_mask[global_index] = False

    positive_indices = torch.argsort(opsd_advantage, descending=True)[: args.top_contexts].tolist()
    negative_indices = torch.argsort(opsd_advantage)[: args.top_contexts].tolist()
    lines = [
        f"# Signed OPSD token credit report — step {step}",
        "",
        f"- Samples: {len(sample_indices)}",
        f"- Valid response tokens: {opsd_advantage.numel()}",
        f"- OPSD advantage coefficient: {float(loaded[0].get('opsd_advantage_coef', 0.01)):g}",
        f"- OPSD advantage clip: {loaded[0].get('opsd_advantage_clip')}",
        f"- Signed advantage p10 / p50 / p90: {fmt_triplet(quantiles(opsd_advantage))}",
        f"- Teacher gap p10 / p50 / p90: {fmt_triplet(quantiles(gap))}",
        f"- Positive/negative/zero token fractions: "
        f"{float((opsd_advantage > 0).float().mean()):.2%} / "
        f"{float((opsd_advantage < 0).float().mean()):.2%} / "
        f"{float((opsd_advantage == 0).float().mean()):.2%}",
    ]

    if outcomes is not None:
        correct = outcomes > 0.5
        incorrect = ~correct
        lines.extend(
            [
                f"- Mean signed advantage, correct trajectories: {float(opsd_advantage[correct].mean()):+.4f}",
                f"- Mean signed advantage, incorrect trajectories: {float(opsd_advantage[incorrect].mean()):+.4f}",
            ]
        )
    if final_mask.any():
        lines.extend(
            [
                f"- Mean signed advantage, reasoning tokens: {float(opsd_advantage[reasoning_mask].mean()):+.4f}",
                f"- Mean signed advantage, `FINAL:` suffix tokens: {float(opsd_advantage[final_mask].mean()):+.4f}",
                f"- Negative advantage, `FINAL:` suffix tokens: "
                f"{float((opsd_advantage[final_mask] < 0).float().mean()):.2%}",
            ]
        )

    if advantages is not None:
        advantages = advantages.float()
        alignment = float(
            torch.dot(opsd_advantage, advantages)
            / (
                torch.linalg.vector_norm(opsd_advantage)
                * torch.linalg.vector_norm(advantages)
            ).clamp_min(1e-12)
        )
        absolute_credit = opsd_advantage.abs().sum().clamp_min(1e-12)
        positive_mass = float(opsd_advantage[advantages > 0].abs().sum() / absolute_credit)
        negative_mass = float(opsd_advantage[advantages < 0].abs().sum() / absolute_credit)
        opsd_weight = tensors.get("opsd_gradient_weights")
        if opsd_weight is not None:
            opsd_rms = torch.sqrt(torch.mean(opsd_weight.float().square()))
            grpo_rms = torch.sqrt(torch.mean(advantages.square())).clamp_min(1e-12)
            lines.append(f"- OPSD/GRPO gradient RMS proxy: {float(opsd_rms / grpo_rms):.4%}")
        lines.extend(
            [
                f"- OPSD–GRPO advantage cosine: {alignment:+.4f}",
                f"- Absolute OPSD credit on positive-GRPO tokens: {positive_mass:.2%}",
                f"- Absolute OPSD credit on negative-GRPO tokens: {negative_mass:.2%}",
            ]
        )

    def append_contexts(title: str, ranked_indices: list[int]) -> None:
        lines.extend(["", f"## {title}", ""])
        for rank_index, global_index in enumerate(ranked_indices, 1):
            sample_id = int(sample_ids[global_index])
            ordered = sorted(sample_indices[sample_id], key=lambda idx: int(positions[idx]))
            local = ordered.index(global_index)
            left, right = max(0, local - 6), min(len(ordered), local + 7)
            context_ids = [int(token_ids[idx]) for idx in ordered[left:right]]
            context = tokenizer.decode(context_ids, skip_special_tokens=True).replace("\n", "\\n")
            token_text = tokenizer.decode(
                [int(token_ids[global_index])], skip_special_tokens=True
            ).replace("\n", "\\n")
            outcome = None if outcomes is None else bool(outcomes[global_index] > 0.5)
            rollout = rollout_by_id.get(sample_id, {})
            lines.append(
                f"{rank_index}. advantage={float(opsd_advantage[global_index]):+.4f}, "
                f"gap={float(gap[global_index]):+.4f}, token=`{token_text}`, "
                f"position={int(positions[global_index])}, correct={outcome}, "
                f"sample={sample_id}, predicted={rollout.get('predicted_label')} — `{context}`"
            )

    append_contexts("Most positive OPSD token contexts", positive_indices)
    append_contexts("Most negative OPSD token contexts", negative_indices)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
