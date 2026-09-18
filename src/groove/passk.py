"""Question-level pass@k diagnostics for fully scored, independent samples."""

from collections import Counter, defaultdict
from math import comb


def summarize_passk(records, *, n=8, expected_question_ids=None):
    """Require complete binary-scored groups; never count missing samples as failures.

    With exactly n=k samples, pass@k is the fraction of questions with at least
    one correct answer. Smaller k use the standard sampling-without-replacement
    estimator 1 - C(n-c, k) / C(n, k).
    """
    if n < 1:
        raise ValueError("n must be positive")
    groups = defaultdict(list)
    for record in records:
        groups[str(record["question_id"])].append(record)
    if not groups:
        raise ValueError("No scored questions")
    if expected_question_ids is not None and set(groups) != set(map(str, expected_question_ids)):
        raise ValueError("Question coverage does not match the benchmark")
    questions = []
    for question_id, rows in sorted(groups.items()):
        if len(rows) != n or {r["sample_index"] for r in rows} != set(range(n)):
            raise ValueError(f"Question {question_id} needs exactly {n} distinct sample indices")
        for field in ("category", "question", "ground_truth", "image_sha256", "prompt_sha256"):
            if len({r[field] for r in rows}) != 1:
                raise ValueError(f"Question {question_id} has inconsistent {field}")
        scores = [r["accuracy"] for r in rows]
        if any(score not in (0, 1) for score in scores):
            raise ValueError(f"Question {question_id} must have binary semantic accuracy")
        correct = int(sum(scores))
        questions.append({"question_id": question_id, "category": rows[0]["category"],
                          "correct_samples": correct, "samples": n,
                          "pass_at_n": correct > 0,
                          "ground_truth": rows[0]["ground_truth"],
                          "image_sha256": rows[0]["image_sha256"],
                          "prompt_sha256": rows[0]["prompt_sha256"]})

    def aggregate(items):
        return {
            "questions": len(items), "samples": len(items) * n,
            "correct_samples": sum(q["correct_samples"] for q in items),
            "passed_questions": sum(q["pass_at_n"] for q in items),
            "pass_at_k": {
                str(k): sum(1 - (comb(n - q["correct_samples"], k) / comb(n, k)
                                if n - q["correct_samples"] >= k else 0)
                            for q in items) / len(items)
                for k in sorted({1, min(2, n), min(4, n), n})
            },
            "correct_samples_histogram": dict(sorted(Counter(q["correct_samples"] for q in items).items())),
        }

    return {"n": n, "overall": aggregate(questions),
            "categories": {category: aggregate([q for q in questions if q["category"] == category])
                           for category in sorted({q["category"] for q in questions})},
            "per_question": questions}


def compare_passk(base, best):
    """Pair questions and expose gains/losses rather than treating samples as questions."""
    if base["n"] != best["n"]:
        raise ValueError("Both models must have the same sample count")
    left = {q["question_id"]: q for q in base["per_question"]}
    right = {q["question_id"]: q for q in best["per_question"]}
    if set(left) != set(right):
        raise ValueError("Model question sets differ")
    gained, lost, both, neither = [], [], [], []
    for key in sorted(left):
        a, b = left[key], right[key]
        for field in ("category", "ground_truth", "image_sha256", "prompt_sha256"):
            if a[field] != b[field]:
                raise ValueError(f"Model inputs differ for {key}: {field}")
        target = (both if a["pass_at_n"] else gained) if b["pass_at_n"] else (lost if a["pass_at_n"] else neither)
        target.append(key)
    return {"questions": len(left), "n": base["n"],
            "base_pass_at_n": base["overall"]["pass_at_k"][str(base["n"])],
            "best_pass_at_n": best["overall"]["pass_at_k"][str(best["n"])],
            "difference_percentage_points": 100 * (len(gained) - len(lost)) / len(left),
            "gained_question_ids": gained, "lost_question_ids": lost,
            "both_pass_question_ids": both, "neither_pass_question_ids": neither}
