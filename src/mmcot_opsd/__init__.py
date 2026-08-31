"""Multimodal CoT self-evolution with group-contrastive visual OPD."""

from .losses import seed_opd_loss
from .schemas import FocusProgram, GroupRollout, Rollout

__all__ = ["FocusProgram", "GroupRollout", "Rollout", "seed_opd_loss"]

