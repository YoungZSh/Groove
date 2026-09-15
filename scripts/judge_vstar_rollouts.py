#!/usr/bin/env python3
"""Judge V* rollout answers through an OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path


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

COMPLETE_ANSWER_PATTERN = re.compile(r"<answer>.*?</answer>", re.IGNORECASE | re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8002/v1")
    parser.add_argument("--model", default="Qwen3.8-27B")
    parser.add_argument("--api-key-env", default="JUDGE_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--done-marker", type=Path)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--expected-rollouts", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--summary-every-rollouts", type=int, default=512)
    return parser.parse_args()


def answer_text(output: str) -> str:
    """If no answer tag is present, judge the full output."""
    return output.split("<answer>")[-1].split("</answer>")[0].strip()


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


def post_json(url: str, api_key: str, body: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def judge_one(item: dict, args: argparse.Namespace, api_key: str) -> dict:
    answer = answer_text(item["model_output"])
    body = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": judge_prompt(item["question"], item["ground_truth"], answer),
            },
        ],
        "temperature": args.temperature,
        "max_completion_tokens": 16,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    endpoint = args.base_url.rstrip("/") + "/chat/completions"
    error: str | None = None
    for attempt in range(args.max_retries + 1):
        try:
            payload = post_json(endpoint, api_key, body, args.timeout)
            response = str(payload["choices"][0]["message"].get("content") or "").strip()
            return {
                **item,
                "answer_text": answer,
                "has_complete_answer_tag": bool(COMPLETE_ANSWER_PATTERN.search(item["model_output"])),
                "judge_response": response,
                "correct": parse_judgement(response),
                "error": None,
            }
        except (KeyError, ValueError, TimeoutError, urllib.error.URLError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            if attempt < args.max_retries:
                time.sleep(0.5 * (2**attempt))
    return {
        **item,
        "answer_text": answer,
        "has_complete_answer_tag": bool(COMPLETE_ANSWER_PATTERN.search(item["model_output"])),
        "judge_response": None,
        "correct": None,
        "error": error,
    }


def load_items(paths: list[Path], tolerate_partial: bool = False) -> list[dict]:
    items: list[dict] = []
    for path in paths:
        if not path.exists() and tolerate_partial:
            continue
        with path.open(encoding="utf-8") as handle:
            records = []
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    if tolerate_partial:
                        break
                    raise RuntimeError(f"invalid JSON at {path}:{line_number}")
        for record in records:
            for rollout_index, completion in enumerate(record["completions"], start=1):
                items.append(
                    {
                        "index": int(record["index"]),
                        "rollout": rollout_index,
                        "question": record["question"],
                        "ground_truth": record["ground_truth"],
                        "model_output": completion["text"],
                        "finish_reason": completion["finish_reason"],
                    }
                )
    return sorted(items, key=lambda item: (item["index"], item["rollout"]))


def item_key(item: dict) -> tuple[int, int]:
    return int(item["index"]), int(item["rollout"])


def load_existing_results(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def judge_batch(items: list[dict], args: argparse.Namespace, api_key: str) -> list[dict]:
    results: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(judge_one, item, args, api_key) for item in items]
        for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            results.append(future.result())
            if completed % 100 == 0 or completed == len(futures):
                print(f"judged batch {completed}/{len(futures)}", flush=True)
    return sorted(results, key=item_key)


def build_summary(results: list[dict], args: argparse.Namespace, elapsed: float, complete: bool) -> dict:
    groups: dict[int, list[dict]] = {}
    for item in results:
        groups.setdefault(int(item["index"]), []).append(item)
    per_question = []
    for index in sorted(groups):
        group = sorted(groups[index], key=lambda item: item["rollout"])
        valid = [item for item in group if item["correct"] is not None]
        per_question.append(
            {
                "index": index,
                "question": group[0]["question"],
                "ground_truth": group[0]["ground_truth"],
                "correct_count": sum(item["correct"] for item in valid),
                "judged_count": len(valid),
                "missing_tag_count": sum(not item["has_complete_answer_tag"] for item in group),
                "answers": [item["answer_text"] for item in group],
                "judgements": [item["correct"] for item in group],
            }
        )
    return {
        "model": args.model,
        "base_url": args.base_url,
        "temperature": args.temperature,
        "complete": complete,
        "questions": len(per_question),
        "rollouts": len(results),
        "correct_rollouts": sum(item["correct"] or 0 for item in results),
        "judge_errors": sum(item["correct"] is None for item in results),
        "elapsed_seconds": elapsed,
        "per_question": per_question,
    }


def write_summary(summary: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.concurrency <= 0:
        raise ValueError("concurrency must be positive")
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        raise RuntimeError(f"missing API key environment variable: {args.api_key_env}")
    if args.follow and args.done_marker is None:
        raise ValueError("--follow requires --done-marker")
    if args.poll_seconds <= 0:
        raise ValueError("poll-seconds must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.summary_every_rollouts <= 0:
        raise ValueError("summary-every-rollouts must be positive")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    results = load_existing_results(args.output)
    known = {item_key(item) for item in results}
    output_mode = "a" if results else "w"
    complete = False
    last_input_signature: tuple[tuple[int, int] | None, ...] | None = None
    cached_items: list[dict] = []
    last_summary_count = len(results)
    with args.output.open(output_mode, encoding="utf-8") as handle:
        while True:
            input_signature = tuple(
                (path.stat().st_size, path.stat().st_mtime_ns) if path.exists() else None
                for path in args.inputs
            )
            marker_exists = args.done_marker is not None and args.done_marker.exists()
            if input_signature != last_input_signature or not args.follow:
                cached_items = load_items(args.inputs, tolerate_partial=args.follow)
                last_input_signature = input_signature
            pending = [
                item for item in cached_items if item_key(item) not in known
            ][: args.batch_size]
            if pending:
                batch_results = judge_batch(pending, args, api_key)
                for result in batch_results:
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    results.append(result)
                    known.add(item_key(result))
                handle.flush()
                os.fsync(handle.fileno())
                if len(results) - last_summary_count >= args.summary_every_rollouts:
                    summary = build_summary(
                        results,
                        args,
                        time.perf_counter() - started,
                        complete=False,
                    )
                    write_summary(summary, args.summary)
                    last_summary_count = len(results)
                else:
                    summary = {
                        "questions": len({item["index"] for item in results}),
                        "rollouts": len(results),
                        "correct_rollouts": sum(item["correct"] or 0 for item in results),
                        "judge_errors": sum(item["correct"] is None for item in results),
                    }
                print(
                    json.dumps(
                        {
                            "event": "judge_progress",
                            "questions": summary["questions"],
                            "rollouts": summary["rollouts"],
                            "correct_rollouts": summary["correct_rollouts"],
                            "judge_errors": summary["judge_errors"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                continue
            if not args.follow:
                complete = True
                break
            if marker_exists:
                if args.expected_rollouts and len(results) != args.expected_rollouts:
                    raise RuntimeError(
                        f"done marker found with {len(results)} judged rollouts; "
                        f"expected {args.expected_rollouts}"
                    )
                complete = True
                break
            time.sleep(args.poll_seconds)

    results.sort(key=item_key)
    summary = build_summary(results, args, time.perf_counter() - started, complete=complete)
    write_summary(summary, args.summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_question"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
