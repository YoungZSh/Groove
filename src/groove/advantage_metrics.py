"""Batch-level diagnostics for outcome and visual-evidence credit."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import torch


def _distribution(values: torch.Tensor) -> dict[str, float]:
    values = values.float()
    if values.numel() == 0:
        return dict.fromkeys(("count", "mean", "std", "rms", "p10", "p50", "p90"), 0.0)
    p10, p50, p90 = torch.quantile(values, values.new_tensor([0.1, 0.5, 0.9])).tolist()
    return {
        "count": float(values.numel()),
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "rms": float(values.square().mean().sqrt()),
        "p10": p10,
        "p50": p50,
        "p90": p90,
    }


@torch.no_grad()
def compute_advantage_metrics(
    *,
    grpo_advantages: torch.Tensor,
    opsd_advantages: torch.Tensor,
    total_advantages: torch.Tensor,
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    evidence_mask: torch.Tensor,
    opsd_coef: float,
    sequence_rewards: torch.Tensor,
    group_ids: Sequence,
) -> dict[str, float]:
    """Measure credit without using reward or group type to weight the update.

    RMS comparisons use the same valid response tokens, including zero evidence
    credit on fallback rows. Gap and trajectory distributions use evidence rows
    only. Empty subsets report count=0; an undefined RMS ratio reports a separate
    flag instead of a misleading enormous value for uniform-reward batches.
    """
    valid = response_mask.detach().bool()
    ready = evidence_mask.detach().to(device=valid.device, dtype=torch.bool)
    active = valid & ready.unsqueeze(-1)
    gap = teacher_log_probs.to(valid.device).float() - student_log_probs.float()
    gap = torch.where(active, gap, 0.0)
    grpo = grpo_advantages[valid].float()
    opsd = opsd_advantages[valid].float()
    metrics: dict[str, float] = {}

    def record(prefix: str, values: torch.Tensor) -> dict[str, float]:
        stats = _distribution(values)
        metrics.update({f"{prefix}_{name}": value for name, value in stats.items()})
        return stats

    record("opsd/delta", gap[active])
    grpo_stats = record("grpo/advantage", grpo)
    opsd_stats = record("opsd/advantage", opsd)
    record("total_advantage", total_advantages[valid])
    metrics["opsd/advantage_rms_raw"] = opsd_stats["rms"]
    metrics["opsd/advantage_rms_weighted"] = abs(opsd_coef) * opsd_stats["rms"]
    ratio_defined = grpo_stats["rms"] > 0
    metrics["opsd_to_grpo_advantage_rms_ratio_defined"] = float(ratio_defined)
    metrics["opsd_to_grpo_advantage_rms_ratio"] = (
        metrics["opsd/advantage_rms_weighted"] / grpo_stats["rms"] if ratio_defined else 0.0
    )
    norm_product = grpo.norm() * opsd.norm()
    metrics["opsd/grpo_cosine"] = float(torch.dot(grpo, opsd) / norm_product) if norm_product > 0 else 0.0
    both_nonzero = (grpo != 0) & (opsd != 0)
    metrics["opsd/grpo_alignment_token_count"] = float(both_nonzero.sum())
    metrics["opsd/grpo_sign_alignment"] = (
        float((grpo[both_nonzero].sign() == opsd[both_nonzero].sign()).float().mean())
        if both_nonzero.any() else 0.0
    )
    ready_fraction = float(ready.float().mean()) if ready.numel() else 0.0
    metrics["opsd/evidence_ready_fraction"] = ready_fraction
    metrics["opsd/evidence_missing_fraction"] = 1.0 - ready_fraction if ready.numel() else 0.0
    metrics["opsd/fallback_fraction"] = metrics["opsd/evidence_missing_fraction"]

    lengths = valid.sum(-1).clamp_min(1)
    trajectory_delta = gap.sum(-1) / lengths
    trajectory_credit = opsd_advantages.sum(-1)
    active_rows = active.any(-1)
    record("opsd/trajectory_delta", trajectory_delta[active_rows])
    record("opsd/trajectory_credit", trajectory_credit[active_rows])

    # Outcome labels are diagnostic only; they never enter the advantage formula.
    correct = sequence_rewards.detach().to(valid.device) > 0.5
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, uid in enumerate(group_ids):
        grouped[str(uid)].append(index)
    group_masks = {name: torch.zeros_like(correct) for name in ("mixed", "all_correct", "all_wrong")}
    for indices in grouped.values():
        outcomes = correct[indices]
        label = "all_correct" if outcomes.all() else "all_wrong" if not outcomes.any() else "mixed"
        group_masks[label][indices] = True
    for name, rows in {"correct": correct, "incorrect": ~correct, **group_masks}.items():
        tokens = active & rows.unsqueeze(-1)
        evidence_rows = active_rows & rows
        record(f"opsd/{name}/delta", gap[tokens])
        record(f"opsd/{name}/advantage", opsd_advantages[tokens])
        record(f"opsd/{name}/trajectory_delta", trajectory_delta[evidence_rows])
        record(f"opsd/{name}/trajectory_credit", trajectory_credit[evidence_rows])
    return metrics
