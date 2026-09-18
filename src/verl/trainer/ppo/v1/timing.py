"""Timing diagnostics for native V1 training; these never affect sampling or rewards."""

import json
from pathlib import Path

import numpy as np


def agent_loop_timing_metrics(rows, non_padding_mask):
    """Summarize retained trajectories only; overlapping request times are not wall time."""
    if len(rows) != len(non_padding_mask):
        raise ValueError("Agent timing rows and padding mask must have equal lengths")
    result = {}
    for name in ("generate_sequences", "compute_score"):
        values = [float(row[name]) for row, keep in zip(rows, non_padding_mask, strict=True)
                  if keep and isinstance(row, dict) and name in row]
        if values:
            for statistic, value in (("mean", np.mean(values)), ("max", np.max(values)),
                                     ("p95", np.percentile(values, 95))):
                result[f"timing_s/retained_agent/{name}/{statistic}"] = float(value)
    return result


def append_step_timing(directory, step, timings):
    """Append after logging finishes, so even the last step's upload time is retained."""
    if directory is None:
        return
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    record = {"step": int(step), "timing_s": dict(timings)}
    with (folder / "steps.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, allow_nan=False) + "\n")
