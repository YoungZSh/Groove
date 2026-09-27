"""Preserve per-trajectory reward diagnostics when using TransferQueue."""

import numpy as np


def judge_retry_metrics(columns):
    """Count retry-exhausted rollouts, including explicit zeros on healthy steps."""
    metrics = {}
    exhausted = columns.get("judge_retries_exhausted", [])
    flags = [float(value) for value in exhausted if value is not None and np.isfinite(value)]
    if flags:
        metrics["reward/judge_retries_exhausted_count"] = float(sum(flags))
        metrics["reward/judge_retries_exhausted_fraction"] = float(np.mean(flags))
    attempts = columns.get("judge_attempts", [])
    attempts = [float(value) for value in attempts if value is not None and np.isfinite(value)]
    if attempts:
        metrics["reward/judge_attempts_mean"] = float(np.mean(attempts))
        metrics["reward/judge_attempts_max"] = float(max(attempts))
    return metrics


def reward_columns(extra_fields, indices=None):
    """Return aligned columns, including None for absent per-row fields."""
    if indices is None:
        indices = range(len(extra_fields))
    rows = []
    for index in indices:
        field = extra_fields[index]
        values = field.get("reward_extra_info", {}) if isinstance(field, dict) else {}
        rows.append(values if isinstance(values, dict) else {})
    keys = sorted({key for row in rows for key in row})
    return {key: [row.get(key) for row in rows] for key in keys}


def reward_extra_metrics(extra_fields, valid_mask):
    """Average finite scalar diagnostics across non-padding trajectories."""
    if len(extra_fields) != len(valid_mask):
        raise ValueError("Reward diagnostics and padding mask must align")
    columns = reward_columns(extra_fields, np.flatnonzero(valid_mask))
    metrics = {}
    for key, values in columns.items():
        numeric = [float(value) for value in values if isinstance(value, (int, float, bool, np.number))]
        numeric = [value for value in numeric if np.isfinite(value)]
        if numeric:
            metrics[f"reward/{key}_mean"] = float(np.mean(numeric))
    metrics.update(judge_retry_metrics(columns))
    return metrics
