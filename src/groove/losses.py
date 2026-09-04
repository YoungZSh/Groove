"""Uncentered signed OPSD advantages, independent of verl internals."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SignedOPSDMetrics:
    advantage_mean: float
    advantage_std: float
    advantage_rms: float
    positive_token_fraction: float
    negative_token_fraction: float
    zero_token_fraction: float
    teacher_gap_mean: float
    active_token_fraction: float
    clipped_token_fraction: float


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denominator = mask.sum().clamp_min(1.0)
    return (values * mask).sum() / denominator


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

    if student_log_prob.shape != teacher_log_prob.shape:
        raise ValueError("Student and teacher log-probability shapes must match")
    if student_log_prob.shape != response_mask.shape:
        raise ValueError("response_mask must match log-probability shape")
    if advantage_clip is not None and advantage_clip <= 0:
        raise ValueError("advantage_clip must be positive when set")

    mask = response_mask.to(dtype=student_log_prob.dtype)
    if evidence_mask is not None:
        if evidence_mask.ndim != 1 or evidence_mask.shape[0] != mask.shape[0]:
            raise ValueError("evidence_mask must have shape [batch]")
        mask = mask * evidence_mask.to(device=mask.device, dtype=mask.dtype).unsqueeze(-1)

    gap = (teacher_log_prob.detach() - student_log_prob.detach()).detach()
    advantages = gap
    if advantage_clip is not None:
        advantages = advantages.clamp(min=-float(advantage_clip), max=float(advantage_clip))
    advantages = (advantages * mask).detach()

    active = mask > 0
    total_tokens = torch.tensor(mask.numel(), device=mask.device, dtype=mask.dtype)
    if not active.any():
        zero = 0.0
        metrics = SignedOPSDMetrics(
            advantage_mean=zero,
            advantage_std=zero,
            advantage_rms=zero,
            positive_token_fraction=zero,
            negative_token_fraction=zero,
            zero_token_fraction=zero,
            teacher_gap_mean=zero,
            active_token_fraction=zero,
            clipped_token_fraction=zero,
        )
        return advantages, metrics

    valid_advantages = advantages[active].float()
    valid_gap = gap[active].float()
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
        teacher_gap_mean=float(_masked_mean(gap, mask)),
        active_token_fraction=float(active.sum().to(mask.dtype).div(total_tokens)),
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
    if opsd_coef < 0:
        raise ValueError("opsd_coef must be non-negative")
    return grpo_advantages.detach() + float(opsd_coef) * opsd_advantages.detach()
