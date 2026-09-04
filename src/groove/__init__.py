"""GROOVE: Group-Relative On-Policy Optimization via Visual Evidence."""

from .losses import combine_grpo_opsd_advantages, groove_opsd_advantages
from .schemas import FocusProgram, GroupRollout, Rollout

__all__ = [
    "FocusProgram",
    "GroupRollout",
    "Rollout",
    "combine_grpo_opsd_advantages",
    "groove_opsd_advantages",
]
