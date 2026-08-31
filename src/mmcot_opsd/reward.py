"""Rule-based reward for multiple-choice VQA rollouts.

The terminal score combines an answer reward with a small incentive for a
machine-readable final line.  The latter lets downstream rollout analysis rely
on one unambiguous answer rather than re-parsing free-form reasoning.
"""

from __future__ import annotations

import re
from typing import Any


FINAL_PATTERNS = (
    re.compile(r"(?i)FINAL\s*(?:ANSWER)?\s*[:：]\s*\(?([A-D])\)?"),
    re.compile(r"(?i)(?:ANSWER|OPTION)\s*(?:IS)?\s*[:：]?\s*\(?([A-D])\)?"),
    re.compile(r"(?<![A-Z])\(([A-D])\)(?![A-Z])"),
)
# ``X`` is deliberately unrestricted. ``FINAL: X`` is a generic extraction
# protocol; the answer comparator decides how X is matched to a dataset label.
FINAL_ANSWER_PATTERN = re.compile(r"(?is)\bFINAL\s*:\s*([^\r\n]*\S[^\r\n]*)\s*\Z")

ANSWER_REWARD_WEIGHT = 0.9
FORMAT_REWARD_WEIGHT = 0.1


def extract_option(text: str) -> str | None:
    for pattern in FINAL_PATTERNS:
        matches = pattern.findall(text or "")
        if matches:
            return matches[-1].upper()
    stripped = (text or "").strip().upper()
    return stripped if stripped in {"A", "B", "C", "D"} else None


def extract_final_answer(text: str) -> str | None:
    """Return X only when the terminal response suffix is ``FINAL: X``.

    X is any nonempty, single-line answer. Matching ``FINAL`` is
    case-insensitive so harmless capitalization differences do not affect the
    extraction protocol.
    """
    match = FINAL_ANSWER_PATTERN.search(text or "")
    return match.group(1).strip() if match else None


def extract_final_option(text: str) -> str | None:
    """Extract a multiple-choice label from a generic terminal final answer."""
    final_answer = extract_final_answer(text)
    return extract_option(final_answer) if final_answer is not None else None


def format_reward(solution_str: str) -> float:
    """Give one point only for a terminal, extractable ``FINAL: X`` suffix."""
    return float(extract_final_answer(solution_str) is not None)


def _normalize_answer(value: str) -> str:
    normalized = " ".join(value.strip().split())
    # Markdown wrappers are presentation, not part of a short final answer.
    while (normalized.startswith("**") and normalized.endswith("**")) or (
        normalized.startswith("__") and normalized.endswith("__")
    ):
        normalized = normalized[2:-2].strip()
    return normalized.casefold()


def answers_match(predicted_answer: str | None, ground_truth: str) -> bool:
    """Compare generic finals, while preserving flexible MCQ letter matching."""
    if predicted_answer is None:
        return False
    gold = str(ground_truth).strip()
    if gold.upper() in {"A", "B", "C", "D"}:
        return extract_option(predicted_answer) == gold.upper()
    return _normalize_answer(predicted_answer) == _normalize_answer(gold)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    answer_reward_weight: float = ANSWER_REWARD_WEIGHT,
    format_reward_weight: float = FORMAT_REWARD_WEIGHT,
    **_: Any,
) -> dict[str, float | str]:
    del data_source, extra_info
    answer_reward_weight = float(answer_reward_weight)
    format_reward_weight = float(format_reward_weight)
    if answer_reward_weight < 0 or format_reward_weight < 0:
        raise ValueError("Terminal reward weights must be non-negative")
    final_answer = extract_final_answer(solution_str)
    # The answer reward is intentionally independent from the format reward:
    # a correct MCQ answer without FINAL still earns 0.9, which makes the 0.1
    # formatting incentive meaningful rather than all-or-nothing.
    predicted = final_answer if final_answer is not None else extract_option(solution_str)
    gold = str(ground_truth).strip().upper()
    answer_reward = float(answers_match(predicted, gold))
    final_format_reward = format_reward(solution_str)
    weighted_answer_reward = answer_reward_weight * answer_reward
    weighted_format_reward = format_reward_weight * final_format_reward
    reward = weighted_answer_reward + weighted_format_reward
    return {
        "score": reward,
        "accuracy": answer_reward,
        "answer_reward": answer_reward,
        "format_reward": final_format_reward,
        "weighted_answer_reward": weighted_answer_reward,
        "weighted_format_reward": weighted_format_reward,
        "answer_reward_weight": answer_reward_weight,
        "format_reward_weight": format_reward_weight,
        # VERL's validation reducer correctly skips strings, but attempts a
        # numeric mean for ``None``. Keep diagnostics present for every
        # sample using an empty string sentinel rather than a nullable value.
        "predicted_label": extract_option(predicted) if predicted is not None else "",
        "final_answer": final_answer or "",
        "ground_truth_label": gold,
    }
