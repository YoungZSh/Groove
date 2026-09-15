"""Shared V*Bench question formatting and deterministic answer scoring."""

from __future__ import annotations

import re


DATA_SOURCE = "vstar_bench"

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
TAG_RE = re.compile(r"</?\s*answer\b[^>]*>", re.IGNORECASE)
CHOICE_RE = re.compile(r"^\(([A-D])\)\s*(.+)$", re.MULTILINE)


def question_text(text: str) -> str:
    # Replace only the source's output-format instruction; retain question/options.
    question = re.sub(
        r"\nAnswer with the option(?:'s)? letter.*$", "", text.strip(), flags=re.I
    )
    return question + "\nReturn the selected option letter inside <answer>...</answer>."


def normalized(text: str) -> str:
    return " ".join(text.strip().strip("*` ").split()).rstrip(".! ").casefold()


def parse_prediction(output: str, choices: dict[str, str]) -> dict:
    """Parse a final answer without looking for arbitrary letters in reasoning.

    Accept an isolated option label, a label followed by its matching option
    text, or an exact unambiguous option text. Ambiguous finals remain unparsed.
    """
    matches = list(ANSWER_RE.finditer(output))
    candidate = matches[-1].group(1).strip() if matches else output.strip()
    tags = TAG_RE.findall(output)
    format_valid = bool(
        len(matches) == 1 and candidate and tags == ["<answer>", "</answer>"]
        and not output[matches[-1].end():].strip()
    )
    # Without a complete tag, only use an explicitly marked final line or the
    # full short response; never guess from letters discussed in the rationale.
    if not matches and "\n" in candidate:
        last = candidate.splitlines()[-1].strip()
        if re.match(r"(?i)^(?:final(?: answer)?|answer)\s*[:：]", last):
            candidate = last
    cleaned = candidate.strip().strip("*` ")
    cleaned = re.sub(r"(?i)^(?:(?:the\s+)?(?:final\s+)?answer|option)\s*(?:is\s*)?[:：]?\s*", "", cleaned)
    label = re.fullmatch(r"\(?([A-D])\)?(?:[.:：\-]\s*|\s+)?(.*?)", cleaned, re.DOTALL)
    prediction = None
    method = "unparsed"
    if label and label.group(1) in choices:
        suffix = label.group(2).strip()
        if not suffix or normalized(suffix) == normalized(choices[label.group(1)]):
            prediction = label.group(1)
            method = "option_label"
    if prediction is None:
        equivalent = [key for key, value in choices.items() if normalized(candidate) == normalized(value)]
        if len(equivalent) == 1:
            prediction = equivalent[0]
            method = "exact_option_text"
    return {
        "answer_text": candidate, "predicted_label": prediction,
        "parse_method": method, "format_valid": format_valid,
        "has_complete_answer_tag": bool(matches),
    }



def compute_validation_score(output: str, ground_truth: str, extra_info: dict) -> dict[str, float]:
    """Report benchmark accuracy separately from answer-format diagnostics."""
    if extra_info.get("split") != "validation":
        raise ValueError("V*Bench is reserved for validation")
    choices = extra_info.get("choices") or {}
    if ground_truth not in choices:
        raise ValueError("V*Bench reference label must occur in the answer choices")
    parsed = parse_prediction(output, choices)
    accuracy = float(parsed["predicted_label"] == ground_truth)
    return {
        "score": accuracy,
        "accuracy": accuracy,
        "answer_reward": accuracy,
        "format_valid": float(parsed["format_valid"]),
        "has_answer_tag": float(parsed["format_valid"]),
        "unparsed": float(parsed["predicted_label"] is None),
        "answer_characters": float(len(parsed["answer_text"])),
    }
