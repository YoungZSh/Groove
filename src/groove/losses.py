"""Uncentered signed OPSD advantages, independent of verl internals."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SignedOPSDMetrics:
    advantage_mean: float = 0.0
    advantage_std: float = 0.0
    advantage_rms: float = 0.0
    positive_token_fraction: float = 0.0
    negative_token_fraction: float = 0.0
    zero_token_fraction: float = 0.0
    teacher_gap_mean: float = 0.0
    delta_std: float = 0.0
    delta_p10: float = 0.0
    delta_p50: float = 0.0
    delta_p90: float = 0.0
    active_token_fraction: float = 0.0
    clipped_token_fraction: float = 0.0


def _binary_mask(mask: torch.Tensor, *, name: str, device: torch.device) -> torch.Tensor:
    mask = mask.detach().to(device=device)
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError(f"{name} must contain only 0 or 1")
    return mask.bool()


def groove_opsd_advantages(
    student_log_prob: torch.Tensor,
    teacher_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    evidence_mask: torch.Tensor | None = None,
    advantage_clip: float | None = None,
) -> tuple[torch.Tensor, SignedOPSDMetrics]:
    """Return detached, uncentered sampled reverse-KL token advantages.

    The evidence advantage is ``log p_teacher(y) - log p_student(y)`` for
    the response token ``y`` sampled by the student.  It is intentionally
    signed and uncentered.  ``evidence_mask`` denotes teacher availability,
    not correctness, so correct and incorrect rollouts use the same rule.
    """

    if student_log_prob.ndim != 2:
        raise ValueError("Sampled-token log probabilities must have shape [batch, tokens]")
    if student_log_prob.shape != teacher_log_prob.shape:
        raise ValueError("Student and teacher log-probability shapes must match")
    if student_log_prob.shape != response_mask.shape:
        raise ValueError("response_mask must match log-probability shape")
    if advantage_clip is not None and (not math.isfinite(advantage_clip) or advantage_clip <= 0):
        raise ValueError("advantage_clip must be finite and positive when set")

    response_valid = _binary_mask(response_mask, name="response_mask", device=student_log_prob.device)
    active = response_valid.clone()
    if evidence_mask is not None:
        if evidence_mask.ndim != 1 or evidence_mask.shape[0] != active.shape[0]:
            raise ValueError("evidence_mask must have shape [batch]")
        active &= _binary_mask(
            evidence_mask, name="evidence_mask", device=active.device
        ).unsqueeze(-1)

    # Score targets are computed before the actor update and remain fixed for
    # all mini-batches. Accumulate the gap in FP32 even for BF16 model outputs.
    gap = (
        teacher_log_prob.detach().to(device=active.device, dtype=torch.float32)
        - student_log_prob.detach().float()
    )
    valid_gap = gap[active]
    if not torch.isfinite(valid_gap).all():
        raise ValueError("Non-finite Teacher/Student log-probability gap on an evidence token")
    # Multiplication by zero would leak NaN/Inf from padding or absent evidence.
    advantages = torch.where(active, gap, 0.0)
    if advantage_clip is not None:
        advantages = advantages.clamp(min=-float(advantage_clip), max=float(advantage_clip))

    if not active.any():
        return advantages, SignedOPSDMetrics()

    valid_advantages = advantages[active]
    p10, p50, p90 = torch.quantile(valid_gap, valid_gap.new_tensor([0.1, 0.5, 0.9])).tolist()
    clipped_fraction = 0.0
    if advantage_clip is not None:
        clipped_fraction = float((valid_gap.abs() > float(advantage_clip)).float().mean())
    metrics = SignedOPSDMetrics(
        advantage_mean=float(valid_advantages.mean()),
        advantage_std=float(valid_advantages.std(unbiased=False)),
        advantage_rms=float(torch.sqrt(valid_advantages.square().mean())),
        positive_token_fraction=float((valid_advantages > 0).float().mean()),
        negative_token_fraction=float((valid_advantages < 0).float().mean()),
        zero_token_fraction=float((valid_advantages == 0).float().mean()),
        teacher_gap_mean=float(valid_gap.mean()),
        delta_std=float(valid_gap.std(unbiased=False)),
        delta_p10=p10,
        delta_p50=p50,
        delta_p90=p90,
        active_token_fraction=float(active.sum() / response_valid.sum().clamp_min(1)),
        clipped_token_fraction=clipped_fraction,
    )
    return advantages, metrics


def combine_grpo_opsd_advantages(
    grpo_advantages: torch.Tensor,
    opsd_advantages: torch.Tensor,
    *,
    opsd_coef: float = 0.01,
) -> torch.Tensor:
    """Add signed OPSD credit to GRPO before the shared PPO loss."""

    if grpo_advantages.shape != opsd_advantages.shape:
        raise ValueError("GRPO and OPSD advantage shapes must match")
    if not math.isfinite(opsd_coef) or opsd_coef < 0:
        raise ValueError("opsd_coef must be finite and non-negative")
    if opsd_coef == 0:
        return grpo_advantages.detach()
    return grpo_advantages.detach() + float(opsd_coef) * opsd_advantages.detach()
