"""Frozen self-teacher snapshots using PyTorch's sharded model-state API.

Each rank retains its own CPU shards. Teacher scoring temporarily installs
these weights in the actor engine, then restores the Student in a finally
block. Optimizer state and the independent reference policy are never copied.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy

import torch
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, set_model_state_dict


def _snapshot(module):
    # Clone even on CPU: state_dict tensors can alias live model storage.
    return deepcopy(get_model_state_dict(module, options=StateDictOptions(cpu_offload=True)))


class FrozenTeacher:
    def __init__(self):
        self.state = None
        self.step = None

    @torch.no_grad()
    def sync(self, module, completed_steps: int, interval: int) -> bool:
        expected = completed_steps // interval * interval
        if self.state is not None and self.step == expected:
            return False
        if completed_steps != expected:
            raise RuntimeError("Missing RLSD Teacher snapshot inside a sync interval; resume from a complete RLSD checkpoint")
        self.state = _snapshot(module)
        self.step = completed_steps
        return True

    @contextmanager
    def apply(self, module):
        if self.state is None:
            raise RuntimeError("RLSD Teacher must be synchronized before scoring")
        student = _snapshot(module)
        was_training = module.training
        try:
            with torch.no_grad():
                set_model_state_dict(module, self.state)
                module.eval()
                yield
        finally:
            with torch.no_grad():
                set_model_state_dict(module, student)
                module.train(was_training)

    def state_dict(self):
        return {"schema_version": 1, "step": self.step, "model": self.state}

    def load_state_dict(self, payload):
        if payload.get("schema_version") != 1:
            raise ValueError("Unsupported RLSD Teacher checkpoint schema")
        self.step, self.state = payload["step"], payload["model"]
