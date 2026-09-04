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


def _judge_one(question: str, ground_truth: str, output: str) -> dict[str, float]:
    base_url = os.environ.get("DEEPEYES_JUDGE_BASE_URL", "http://127.0.0.1:8002/v1").rstrip("/")
    api_key = os.environ.get("DEEPEYES_JUDGE_API_KEY", "")
    model = os.environ.get("DEEPEYES_JUDGE_MODEL", "Qwen3.8-27B")
    timeout = float(os.environ.get("DEEPEYES_JUDGE_TIMEOUT_SECONDS", "180"))
    max_retries = int(os.environ.get("DEEPEYES_JUDGE_MAX_RETRIES", "5"))
    if not api_key:
        raise RuntimeError("DEEPEYES_JUDGE_API_KEY is required")

    answer, has_answer_tag = extract_answer(output)
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": judge_prompt(question, ground_truth, answer),
            },
        ],
        "temperature": 0.0,
        "max_completion_tokens": 16,
        "chat_template_kwargs": {"enable_thinking": False},
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
            return {
                "score": correct,
                "accuracy": correct,
                "answer_reward": correct,
                # Format is diagnostic only and never contributes to score.
                "has_answer_tag": float(has_answer_tag),
                "answer_characters": float(len(answer)),
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
