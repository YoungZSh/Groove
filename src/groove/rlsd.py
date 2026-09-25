"""Positive-advantage RLSD credit, computed once before the PPO update."""

from __future__ import annotations

import math

import torch

from .losses import _binary_mask


def rlsd_lambda(step: int, *, initial: float = 0.5, decay_steps: int = 50) -> float:
    """Linear decay indexed by the logged outer global step (step 50 is zero)."""
    if not math.isfinite(initial) or not 0 <= initial <= 1:
        raise ValueError("RLSD initial lambda must be finite and in [0, 1]")
    if isinstance(decay_steps, bool) or not isinstance(decay_steps, int) or decay_steps <= 0:
        raise ValueError("RLSD decay_steps must be a positive integer")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("RLSD step must be a non-negative integer")
    return initial * max(1.0 - step / decay_steps, 0.0)


@torch.no_grad()
def positive_rlsd_advantages(
    grpo_advantages: torch.Tensor,
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    evidence_mask: torch.Tensor,
    *,
    lam: float,
    clip_range: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Return total advantages, effective weights and diagnostics.

    Only positive GRPO advantages on valid evidence tokens are reweighted.
    Negative/zero advantages and unavailable evidence retain exactly their
    original values. There is no accuracy gate or sequence normalization.
    """
    if not math.isfinite(lam) or not 0 <= lam <= 1:
        raise ValueError("RLSD lambda must be finite and in [0, 1]")
    if not math.isfinite(clip_range) or not 0 < clip_range < 1:
        raise ValueError("RLSD clip_range must be finite and in (0, 1)")
    shape = grpo_advantages.shape
    if len(shape) != 2 or any(t.shape != shape for t in (student_log_probs, teacher_log_probs, response_mask)):
        raise ValueError("RLSD advantages, log probabilities and mask must have matching [batch, tokens] shapes")
    if evidence_mask.shape != (shape[0],):
        raise ValueError("RLSD evidence_mask must have shape [batch]")
    device = grpo_advantages.device
    valid = _binary_mask(response_mask, name="response_mask", device=device)
    ready = _binary_mask(evidence_mask, name="evidence_mask", device=device)
    base = grpo_advantages.detach().float()
    if not torch.isfinite(base[valid]).all():
        raise ValueError("Non-finite GRPO advantage on a response token")
    positive = valid & (base > 0)
    active = positive & ready.unsqueeze(-1) & (lam > 0)
    weights = torch.ones_like(base)
    total = base.clone()
    metrics = {
        "rlsd/lambda": float(lam),
        "rlsd/positive_token_fraction": float(positive.sum() / valid.sum().clamp_min(1)),
        "rlsd/active_token_fraction": float(active.sum() / valid.sum().clamp_min(1)),
        "rlsd/weight_mean": 1.0,
        "rlsd/weight_min": 1.0,
        "rlsd/weight_max": 1.0,
        "rlsd/clip_low_fraction": 0.0,
        "rlsd/clip_high_fraction": 0.0,
        "rlsd/correction_to_grpo_rms_ratio": 0.0,
    }
    if active.any():
        gap = teacher_log_probs.detach().to(device=device, dtype=torch.float32)[active] - student_log_probs.detach().to(
            device=device, dtype=torch.float32
        )[active]
        if not torch.isfinite(gap).all():
            raise ValueError("Non-finite Teacher/Student log-probability gap on an active RLSD token")
        low, high = math.log1p(-clip_range), math.log1p(clip_range)
        # Clip in log space before exp: equivalent weights without overflow.
        clipped_ratio = gap.clamp(low, high).exp()
        weights[active] = 1.0 + lam * (clipped_ratio - 1.0)
        total[active] = base[active] * weights[active]
        metrics.update({
            "rlsd/weight_mean": float(weights[active].mean()),
            "rlsd/weight_min": float(weights[active].min()),
            "rlsd/weight_max": float(weights[active].max()),
            "rlsd/clip_low_fraction": float((gap < low).float().mean()),
            "rlsd/clip_high_fraction": float((gap > high).float().mean()),
        })
        norm = base[valid].norm()
        metrics["rlsd/correction_to_grpo_rms_ratio"] = float((total[valid] - base[valid]).norm() / norm)
    return total, weights, metrics
