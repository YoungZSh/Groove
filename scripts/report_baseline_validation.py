#!/usr/bin/env python3
"""Pair every held-out completion with its pre-training completion for review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import pandas as pd


def write_text(path, text):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def report(run_dir, dataset):
    output_dir = run_dir / "validation_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline = [json.loads(line) for line in (run_dir / "validation/0.jsonl").read_text().splitlines()]
    assert len(baseline) == len(dataset)
    assert len({row["input"] for row in baseline}) == len(baseline), "Ambiguous validation prompt IDs"
    metadata = {}
    for original, source in zip(baseline, dataset.to_dict("records"), strict=True):
        extra = source["extra_info"]
        assert extra["question"] in original["input"], "Validation order differs from dataset"
        assert original["gts"] == source["reward_model"]["ground_truth"]
        metadata[original["input"]] = extra
    baseline_by_input = {row["input"]: row for row in baseline}
    summaries = []
    for path in sorted((run_dir / "validation").glob("*.jsonl"), key=lambda p: int(p.stem)):
        # The trainer writes validation dumps in place; skip incomplete files.
        try:
            rows = [json.loads(line) for line in path.read_text().splitlines()]
        except json.JSONDecodeError:
            continue
        if len(rows) != len(baseline):
            continue
        assert len({row["input"] for row in rows}) == len(rows)
        assert {row["input"] for row in rows} == set(baseline_by_input)
        pairs = []
        for current in rows:
            previous = baseline_by_input[current["input"]]
            extra = metadata[current["input"]]
            assert current["gts"] == previous["gts"]
            before, after = bool(previous["accuracy"]), bool(current["accuracy"])
            pairs.append({
                "question_id": extra["question_id"], "question": extra["question"],
                "image_path": extra["image_path"], "ground_truth": current["gts"],
                "transition": f"{int(before)}->{int(after)}",
                "before": previous, "after": current,
                "reasoning_review": {
                    "status": "unreviewed", "object_localization": None,
                    "visual_claims": None, "evidence_to_conclusion": None,
                    "internal_contradiction": None,
                },
            })
        step = int(path.stem)
        summary = {
            "step": step, "samples": len(rows),
            "correct": sum(bool(row["accuracy"]) for row in rows),
            "accuracy": sum(row["accuracy"] for row in rows) / len(rows),
            "format_rate": sum(row["format_reward"] for row in rows) / len(rows),
            "mean_reward": sum(row["score"] for row in rows) / len(rows),
            "wrong_to_right": sum(p["transition"] == "0->1" for p in pairs),
            "right_to_wrong": sum(p["transition"] == "1->0" for p in pairs),
            "reasoning_reviewed": 0,
        }
        summaries.append(summary)
        pair_path = output_dir / f"step_{step}.jsonl"
        if not pair_path.exists():
            write_text(pair_path, "".join(json.dumps(pair, ensure_ascii=False) + "\n" for pair in pairs))
            lines = [f"# Baseline 完整验证输出：step 0 → {step}", "",
                     "所有验证题均保留。答案得分不代表正文推理正确，语义审核尚未完成。", ""]
            for pair in pairs:
                lines += [f"## {pair['question_id']} · {pair['transition']}", "",
                          pair["question"], "", f"[原图]({pair['image_path']})", "",
                          "训练前完整输出：", "", "> " + pair["before"]["output"].replace("\n", "\n> "),
                          "", "当前完整输出：", "", "> " + pair["after"]["output"].replace("\n", "\n> "), ""]
            write_text(output_dir / f"step_{step}.md", "\n".join(lines))
    write_text(output_dir / "summary.json", json.dumps(summaries, ensure_ascii=False, indent=2) + "\n")
    lines = ["# Baseline 固定验证集", "",
             "每次使用相同原图、题目和贪心解码；答案分数按现有 reward 解析器计算，受 FINAL 格式要求影响。",
             "正文语义判断必须另外结合原图核验，不能从答案变化自动推断。", "",
             "| Step | 正确/总数 | 答案准确率 | 格式通过率 | 错→对 | 对→错 |",
             "|---:|---:|---:|---:|---:|---:|"]
    for row in summaries:
        lines.append(f"| {row['step']} | {row['correct']}/{row['samples']} | {row['accuracy']:.2%} | "
                     f"{row['format_rate']:.2%} | {row['wrong_to_right']} | {row['right_to_wrong']} |")
    write_text(output_dir / "report.md", "\n".join(lines) + "\n")
    return [row["step"] for row in summaries]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--watch-pid", type=int)
    args = parser.parse_args()
    dataset = pd.read_parquet(args.validation_data)
    if args.watch_pid is None:
        print(report(args.run_dir, dataset))
        return
    import psutil

    process = psutil.Process(args.watch_pid)
    previous_steps = None
    while True:
        steps = report(args.run_dir, dataset)
        if steps != previous_steps:
            print(f"Validation comparisons saved for steps {steps}", flush=True)
            previous_steps = steps
        if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
            break
        time.sleep(30)


if __name__ == "__main__":
    main()
