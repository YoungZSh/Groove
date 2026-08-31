"""SEED-style sampled-token OPD loss, independent of verl internals."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SeedOPDMetrics:
    loss: float
    gate_mean: float
    gate_gt_half_fraction: float
    teacher_gap_mean: float
    active_token_fraction: float


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denominator = mask.sum().clamp_min(1.0)
    return (values * mask).sum() / denominator


def seed_opd_loss(
    student_log_prob: torch.Tensor,
    teacher_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    beta: float = 5.0,
    sample_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, SeedOPDMetrics]:
    """Compute SEED Eq. (1) without a reward-derived sample mask.

    ``teacher_log_prob`` and the sigmoid gate are detached. Gradients therefore
    flow only through the ordinary student branch. ``sample_mask`` is reserved
    for evidence availability (for example a group where grounding failed), not
    for correctness; successful and failed rollouts use the same OPD rule.
    """

    if student_log_prob.shape != teacher_log_prob.shape:
        raise ValueError("Student and teacher log-probability shapes must match")
    if student_log_prob.shape != response_mask.shape:
        raise ValueError("response_mask must match log-probability shape")
    if beta <= 0:
        raise ValueError("beta must be positive")

    mask = response_mask.to(dtype=student_log_prob.dtype)
    if sample_mask is not None:
        if sample_mask.ndim != 1 or sample_mask.shape[0] != mask.shape[0]:
            raise ValueError("sample_mask must have shape [batch]")
        mask = mask * sample_mask.to(device=mask.device, dtype=mask.dtype).unsqueeze(-1)

    detached_teacher = teacher_log_prob.detach()
    gap = (detached_teacher - student_log_prob.detach()).detach()
    gate = torch.sigmoid(float(beta) * gap).detach()
    per_token = gate * (detached_teacher - student_log_prob)
    loss = _masked_mean(per_token, mask)

    active = mask > 0
    total_tokens = torch.tensor(mask.numel(), device=mask.device, dtype=mask.dtype)
    metrics = SeedOPDMetrics(
        loss=float(loss.detach()),
        gate_mean=float(_masked_mean(gate, mask).detach()),
        gate_gt_half_fraction=float(_masked_mean((gate > 0.5).to(mask.dtype), mask).detach()),
        teacher_gap_mean=float(_masked_mean(gap, mask).detach()),
        active_token_fraction=float(active.sum().to(mask.dtype).div(total_tokens).detach()),
    )
    return loss, metrics


def joint_visual_seed_loss(
    grpo_loss: torch.Tensor,
    student_log_prob: torch.Tensor,
    teacher_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    opd_coef: float = 0.01,
    beta: float = 5.0,
    evidence_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, SeedOPDMetrics]:
    if opd_coef < 0:
        raise ValueError("opd_coef must be non-negative")
    opd_loss, metrics = seed_opd_loss(
        student_log_prob,
        teacher_log_prob,
        response_mask,
        beta=beta,
        sample_mask=evidence_mask,
    )
    return grpo_loss + float(opd_coef) * opd_loss, opd_loss, metrics

