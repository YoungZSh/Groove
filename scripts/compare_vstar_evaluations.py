#!/usr/bin/env python3
"""Compare two complete V*Bench evaluations with identical inference settings."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load(directory: Path):
    manifest = json.loads((directory / "manifest.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    rows = [json.loads(line) for line in (directory / "predictions.jsonl").read_text().splitlines() if line.strip()]
    assert len(rows) == manifest["rows"] == 191
    records = {row["question_id"]: row for row in rows}
    assert len(records) == len(rows)
    assert sum(row["correct"] for row in rows) == summary["overall"]["correct"]
    return manifest, summary, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grpo", type=Path, required=True)
    parser.add_argument("--opsd", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gm, gs, gr = load(args.grpo)
    om, os, op = load(args.opsd)
    differing = {key for key in gm.keys() | om.keys() if gm.get(key) != om.get(key)}
    assert differing <= {"model", "checkpoint_source", "training_run_id"}, differing
    assert gr.keys() == op.keys()
    pairs = []
    for qid in sorted(gr, key=int):
        g, o = gr[qid], op[qid]
        for key in ("question", "choices", "ground_truth", "image_sha256", "image_size", "prompt", "category", "prompt_tokens"):
            assert g[key] == o[key], (qid, key)
        pairs.append({
            "question_id": qid, "category": g["category"], "ground_truth": g["ground_truth"],
            "grpo_prediction": g["predicted_label"], "opsd_prediction": o["predicted_label"],
            "grpo_correct": g["correct"], "opsd_correct": o["correct"],
        })
    improvements = [p["question_id"] for p in pairs if p["opsd_correct"] and not p["grpo_correct"]]
    regressions = [p["question_id"] for p in pairs if p["grpo_correct"] and not p["opsd_correct"]]
    result = {
        "protocol_identical": True, "checkpoint_step": gm["checkpoint_step"],
        "training_run_ids": [gm["training_run_id"], om["training_run_id"]],
        "grpo": gs, "grpo_opsd": os,
        "accuracy_delta_pp": 100 * (os["overall"]["accuracy"] - gs["overall"]["accuracy"]),
        "improved_question_ids": improvements, "regressed_question_ids": regressions,
        "both_correct": sum(p["grpo_correct"] and p["opsd_correct"] for p in pairs),
        "both_wrong": sum(not p["grpo_correct"] and not p["opsd_correct"] for p in pairs),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "comparison.json").open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    with (args.output / "paired_predictions.csv").open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    lines = [
        "# V*Bench：纯 GRPO 与 GRPO + OPSD", "",
        f"Checkpoint：昨天画图对比的两条 Run，均为 Step {gm['checkpoint_step']}。",
        f"GRPO Run ID：`{gm['training_run_id']}`；GRPO + OPSD Run ID：`{om['training_run_id']}`。", "",
        "每模型单独一张 GPU，vLLM，temperature=0，每题一次生成，最多 1024 response tokens。",
        "使用源 parquet 的原始图像，由 checkpoint 原生 processor 处理；两边的图像、Prompt、模板、解码参数和评分规则已逐项核对一致。",
        "使用完整 191 题 test split；选择题按独立于标签的规则提取答案，不调用 LLM Judge。无法解析的答案计错，Format Penalty 不进入 benchmark accuracy。", "",
        "| 类别 | 纯 GRPO | GRPO + OPSD | 差值 |", "|---|---:|---:|---:|",
    ]
    for title, key in [("总体", None), ("属性识别", "direct_attributes"), ("空间关系", "relative_position")]:
        a = gs["overall"] if key is None else gs["categories"][key]
        b = os["overall"] if key is None else os["categories"][key]
        lines.append(f"| {title} | {a['correct']}/{a['count']} ({a['accuracy']:.2%}) | {b['correct']}/{b['count']} ({b['accuracy']:.2%}) | {(b['accuracy']-a['accuracy'])*100:+.2f} pp |")
    lines += ["", f"逐题比较：{len(improvements)} 题由错变对，{len(regressions)} 题由对变错。", "",
              "| 输出诊断 | 纯 GRPO | GRPO + OPSD |", "|---|---:|---:|"]
    for title, key in [("格式有效", "format_valid"), ("无法解析", "unparsed"), ("达到长度上限", "length_truncated")]:
        lines.append(f"| {title} | {gs['overall'][key]} | {os['overall'][key]} |")
    lines += ["", "## 无法解析的输出", ""]
    for label, records in [("GRPO", gr), ("GRPO + OPSD", op)]:
        for qid, row in records.items():
            if row["predicted_label"] is None:
                lines += [f"### {label} · Q{qid}", "", row["question"], "", "~~~~text", row["output"], "~~~~", ""]
    with (args.output / "report.md").open("x") as f:
        f.write("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
