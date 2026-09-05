"""Batched semantic reward for DeepEyes-style visual QA.
The policy receives only the raw image and question.  A remote text-only judge
compares the policy's final answer with the private reference answer and emits
an independent binary reward for every rollout.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


JUDGE_INSTRUCTION = """Below are two answers to a question. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question. Determine whether these two answers are consistent.

Note that [Model Answer] is consistent with [Standard Answer] whenever they are essentially the same. If the meaning is expressed in the same way, it is considered consistent, for example, 'pink' and 'it is pink'.
If they are consistent, Judgement is 1; if they are different, Judgement is 0. Just output Judgement and don't output anything else.
"""

JUDGE_EXAMPLES = """
[Question]: Is the countertop tan or blue?
[Standard Answer]: The countertop is tan.
[Model_answer]: tan
Judgement: 1

[Question]: On which side of the picture is the barrier?
[Standard Answer]: The barrier is on the left side of the picture.
[Model_answer]: left
Judgement: 1

[Question]: Is the kite brown and large?
[Standard Answer]: Yes, the kite is brown and large.
[Model_answer]: Yes
Judgement: 1

[Question]: Are the spots on a giraffe?
[Standard Answer]: No, the spots are on a banana.
[Model_answer]: no
Judgement: 1

[Question]: Who is wearing pants?
[Standard Answer]: The boy is wearing pants.
[Model_answer]: The person in the picture is wearing pants.
Judgement: 1

[Question]: Is the man phone both blue and closed?
[Standard Answer]: Yes, the man phone is both blue and closed.
[Model_answer]: No.
Judgement: 0

[Question]: What color is the towel in the center of the picture?
[Standard Answer]: The towel in the center of the picture is blue.
[Model_answer]: The towel in the center of the picture is pink.
Judgement: 0
"""

ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class RepetitionHit:
    start: int
    end: int
    period: int
    repeats: int

    @property
    def total_characters(self) -> int:
        return self.period * self.repeats


def extract_answer(output: str) -> tuple[str, bool]:
    """Extract the last complete answer tag, falling back to the full output."""
    matches = ANSWER_PATTERN.findall(output or "")
    if matches:
        return matches[-1].strip(), True
    return (output or "").strip(), False


def judge_prompt(question: str, ground_truth: str, answer: str) -> str:
    return (
        JUDGE_INSTRUCTION
        + JUDGE_EXAMPLES
        + "\n"
        + f"[Question]: {question}\n"
        + f"[Standard Answer]: {ground_truth}\n"
        + f"[Model_answer]: {answer}\n"
        + "Judgement:"
    )


def parse_judgement(response: str) -> int:
    tail = response.rsplit("Judgement:", 1)[-1].strip()
    match = re.search(r"(?<!\d)([01])(?!\d)", tail)
    if not match:
        raise ValueError(f"invalid judge response: {response!r}")
    return int(match.group(1))


def _verify_repetition_at(
    text: str,
    *,
    start: int,
    period: int,
    min_repeats: int,
    min_total_characters: int,
) -> RepetitionHit | None:
    if period <= 0 or start < 0 or start + period > len(text):
        return None

    pattern = text[start : start + period]
    repeats = 0
    end = start
    while end + period <= len(text) and text[end : end + period] == pattern:
        repeats += 1
        end += period

    previous = start - period
    while previous >= 0 and text[previous : previous + period] == pattern:
        repeats += 1
        start = previous
        previous -= period

    if repeats < min_repeats or repeats * period < min_total_characters:
        return None
    return RepetitionHit(start=start, end=end, period=period, repeats=repeats)


def find_inner_repetition(
    output: str,
    *,
    min_repeats: int = 4,
    min_total_characters: int = 80,
    min_period: int = 1,
    max_period: int = 1024,
    sample_length: int = 16,
    sample_interval: int = 32,
) -> RepetitionHit | None:
    """Find an exact contiguous repeated span using an Antidoom-style scan.

    Fingerprints propose candidate periods, after which the full repeated span
    is verified exactly. This detects variable-length loops without assigning a
    global n-gram diversity score to otherwise valid reasoning.
    """
    if min_repeats < 2:
        raise ValueError("repetition minimum repeats must be at least 2")
    if min_total_characters <= 0:
        raise ValueError("repetition minimum total characters must be positive")
    if min_period <= 0 or max_period < min_period:
        raise ValueError("repetition period bounds are invalid")
    if sample_length <= 0 or sample_interval <= 0:
        raise ValueError("repetition fingerprint settings must be positive")

    text = output or ""
    if len(text) < min_total_characters:
        return None

    for sample_start in range(0, len(text) - sample_length + 1, sample_interval):
        fingerprint = text[sample_start : sample_start + sample_length]

        candidate = text.find(fingerprint, sample_start + 1)
        while candidate >= 0:
            period = candidate - sample_start
            if period > max_period:
                break
            if period >= min_period:
                hit = _verify_repetition_at(
                    text,
                    start=sample_start,
                    period=period,
                    min_repeats=min_repeats,
                    min_total_characters=min_total_characters,
                )
                if hit is not None:
                    return hit
            candidate = text.find(fingerprint, candidate + 1)

        candidate = text.rfind(fingerprint, 0, sample_start)
        while candidate >= 0:
            period = sample_start - candidate
            if min_period <= period <= max_period:
                hit = _verify_repetition_at(
                    text,
                    start=candidate,
                    period=period,
                    min_repeats=min_repeats,
                    min_total_characters=min_total_characters,
                )
                if hit is not None:
                    return hit
            candidate = text.rfind(fingerprint, 0, candidate)
    return None


def _configured_repetition_hit(output: str) -> RepetitionHit | None:
    enabled = os.environ.get("DEEPEYES_REPETITION_ZERO_REWARD", "false").strip().lower()
    if enabled not in {"true", "false"}:
        raise ValueError("DEEPEYES_REPETITION_ZERO_REWARD must be true or false")
    if enabled == "false":
        return None

    return find_inner_repetition(
        output,
        min_repeats=int(os.environ.get("DEEPEYES_REPETITION_MIN_REPEATS", "4")),
        min_total_characters=int(os.environ.get("DEEPEYES_REPETITION_MIN_TOTAL_CHARACTERS", "80")),
        min_period=int(os.environ.get("DEEPEYES_REPETITION_MIN_PERIOD", "1")),
        max_period=int(os.environ.get("DEEPEYES_REPETITION_MAX_PERIOD", "1024")),
        sample_length=int(os.environ.get("DEEPEYES_REPETITION_SAMPLE_LENGTH", "16")),
        sample_interval=int(os.environ.get("DEEPEYES_REPETITION_SAMPLE_INTERVAL", "32")),
    )


def _judge_one(question: str, ground_truth: str, output: str) -> dict[str, float]:
    base_url = os.environ.get("DEEPEYES_JUDGE_BASE_URL", "http://127.0.0.1:8002/v1").rstrip("/")
    api_key = os.environ.get("DEEPEYES_JUDGE_API_KEY", "")
    model = os.environ.get("DEEPEYES_JUDGE_MODEL", "Qwen3.8-27B")
    timeout = float(os.environ.get("DEEPEYES_JUDGE_TIMEOUT_SECONDS", "180"))
    max_retries = int(os.environ.get("DEEPEYES_JUDGE_MAX_RETRIES", "5"))
    if not api_key:
        raise RuntimeError("DEEPEYES_JUDGE_API_KEY is required")

    answer, has_answer_tag = extract_answer(output)
    repetition_hit = _configured_repetition_hit(output)
    severe_repetition = repetition_hit is not None
    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a binary semantic-equivalence judge. Output exactly 0 or 1.",
            },
            {
                "role": "user",
                "content": judge_prompt(question, ground_truth, answer),
            },
        ],
        "temperature": 0.0,
        "max_completion_tokens": 4,
        "chat_template_kwargs": {"enable_thinking": False},
        # The local vLLM Judge supports constrained decoding. Without this,
        # an occasional explanatory preamble can consume the completion budget
        # before the model emits its binary decision and abort an entire run.
        "structured_outputs": {"choice": ["0", "1"]},
    }
    request = urllib.request.Request(
        base_url + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.load(response)
            content = str(payload["choices"][0]["message"].get("content") or "").strip()
            correct = float(parse_judgement(content))
            rewarded = 0.0 if severe_repetition else correct
            return {
                "score": rewarded,
                # Preserve semantic accuracy for diagnosis while gating only
                # the reward consumed by GRPO/OPSD.
                "accuracy": correct,
                "answer_reward": rewarded,
                # Format is diagnostic only and never contributes to score.
                "has_answer_tag": float(has_answer_tag),
                "answer_characters": float(len(answer)),
                "severe_repetition": float(severe_repetition),
                "repetition_zeroed_reward": float(severe_repetition and correct > 0.0),
                "repetition_start_character": float(repetition_hit.start if repetition_hit else -1),
                "repetition_period_characters": float(repetition_hit.period if repetition_hit else 0),
                "repetition_count": float(repetition_hit.repeats if repetition_hit else 0),
                "repetition_total_characters": float(
                    repetition_hit.total_characters if repetition_hit else 0
                ),
            }
        except (KeyError, ValueError, TimeoutError, urllib.error.URLError) as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(0.5 * (2**attempt))
    raise RuntimeError(f"remote DeepEyes judge failed after retries: {last_error}")


def compute_score_batched(
    data_sources,
    solution_strs,
    ground_truths,
    extra_infos,
    **_: Any,
) -> list[dict[str, float]]:
    """Score a rollout batch concurrently while preserving input order."""
    del data_sources
    count = len(solution_strs)
    if not (len(ground_truths) == len(extra_infos) == count):
        raise ValueError("batched reward inputs must have equal lengths")
    concurrency = int(os.environ.get("DEEPEYES_JUDGE_CONCURRENCY", "128"))
    if concurrency <= 0:
        raise ValueError("DEEPEYES_JUDGE_CONCURRENCY must be positive")

    questions = [str((info or {}).get("question", "")) for info in extra_infos]
    if any(not question for question in questions):
        raise ValueError("every DeepEyes reward item requires extra_info.question")

    scores: list[dict[str, float] | None] = [None] * count
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(concurrency, count)) as executor:
        future_to_index = {
            executor.submit(
                _judge_one,
                questions[index],
                str(ground_truths[index]),
                str(solution_strs[index]),
            ): index
            for index in range(count)
        }
        for future in concurrent.futures.as_completed(future_to_index):
            scores[future_to_index[future]] = future.result()
    if any(score is None for score in scores):
        raise RuntimeError("remote judge returned an incomplete reward batch")
    return [score for score in scores if score is not None]


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any],
    **_: Any,
) -> dict[str, float]:
    """VERL 0.9 reward-loop adapter for one streamed rollout."""
    del data_source
    question = str((extra_info or {}).get("question", "")).strip()
    if not question:
        raise ValueError("DeepEyes reward requires extra_info.question")
    return _judge_one(question, str(ground_truth), str(solution_str))
