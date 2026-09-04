#!/usr/bin/env python3
"""Render whole rollout sequences and contiguous reasoning units for review."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    data = torch.load(args.audit_dir / f"{args.step}.rank0.pt", map_location="cpu", weights_only=True)
    metadata = [json.loads(line) for line in (args.audit_dir / f"{args.step}.jsonl").read_text().splitlines()]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    units = []
    lines = [
        f"# Baseline 完整推理轨迹 — step {args.step}", "",
        f"{len(metadata)} 条完整轨迹；{len(data['token_ids'])} 个有效 token，含 EOS。",
        "",
        "分数来自本次更新前同一 actor 的两种输入。Teacher 使用原图、裁剪图、focus 和原题。",
        "单个 token、句子和整条轨迹均完整保留；不按 advantage 大小挑选位置。",
        "句界划分仅辅助阅读，语义角色和事实对错需要结合原图与完整上下文核验。",
        "不能用最终答案正确替代正文正确，也不能把短语 A 之和当作真实参数梯度。",
        "",
        "检查顺序：对象定位 → 属性/数量/关系判断 → 证据与结论的连接 → 前后矛盾与错误传播。",
        "需要同时比较训练前后固定验证题的完整输出，判断推理表现是否真正改善。",
    ]
    keys = ["student_log_probs", "teacher_log_probs", "grpo_advantages", "opsd_advantages",
            "weighted_opsd_advantages", "total_advantages"]
    for sample in metadata:
        sample_id = sample["rollout_sample_id"]
        indices = (data["sample_ids"] == sample_id).nonzero().flatten().tolist()
        indices.sort(key=lambda i: int(data["response_positions"][i]))
        sample_rows = []
        current = []
        sample_units = []
        for i in indices:
            token = tokenizer.decode([int(data["token_ids"][i])], skip_special_tokens=False,
                                     clean_up_tokenization_spaces=False)
            row = dict(step=args.step, sample_id=sample_id, question_id=sample["question_id"],
                       uid=sample["uid"], position=int(data["response_positions"][i]),
                       token_id=int(data["token_ids"][i]), token=token,
                       evidence_available=sample["evidence_available"], score=sample["score"])
            row.update({key: float(data[key][i]) for key in keys})
            rows.append(row)
            sample_rows.append(row)
            current.append(row)
            if token.strip() in {".", "!", "?", ";", "<|im_end|>"} or "\n" in token:
                sample_units.append(current)
                current = []
        if current:
            sample_units.append(current)
        assert sum(len(unit) for unit in sample_units) == len(indices)
        lines += ["", f"## Sample {sample_id} · {sample['question_id']}", "",
                  f"Group: `{sample['uid']}` · reward={sample['score']:.3f} · evidence={sample['evidence_available']}",
                  "", str(sample["question"]), "",
                  f"[原图]({sample['image_path']})", "",
                  "> " + sample["output"].replace("\n", "\n> "), "",
                  "| 连续位置 | 完整语句片段 | OPSD 和 | OPSD 均值 | 正/负/零 token |",
                  "|---|---|---:|---:|---|"]
        for index, unit in enumerate(sample_units):
            ids = [r["token_id"] for r in unit]
            text = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            values = [r["opsd_advantages"] for r in unit]
            total = sum(values)
            positive = sum(x > .01 for x in values)
            negative = sum(x < -.01 for x in values)
            near_zero = len(values) - positive - negative
            units.append(dict(step=args.step, sample_id=sample_id, question_id=sample["question_id"],
                              unit_id=index, start=unit[0]["position"], end=unit[-1]["position"],
                              text=text, opsd_sum=total, opsd_mean=total / len(unit),
                              semantic_role="unreviewed", factual_verdict="unreviewed"))
            visible = text.replace("|", "\\|").replace("\n", "\\n")
            lines.append(f"| {unit[0]['position']}–{unit[-1]['position']} | `{visible}` | "
                         f"{total:+.5f} | {total / len(unit):+.5f} | {positive}/{negative}/{near_zero} |")
        lines += ["", "| 位置 | Token | Student log p | Teacher log p | GRPO | OPSD | 组合 A |",
                  "|---:|---|---:|---:|---:|---:|---:|"]
        for row in sample_rows:
            token = row["token"].replace("|", "\\|").replace("\n", "\\n")
            values = [row[k] for k in ["student_log_probs", "teacher_log_probs", "grpo_advantages",
                                      "opsd_advantages", "total_advantages"]]
            lines.append(f"| {row['position']} | `{token}` | " + " | ".join(f"{v:+.6f}" for v in values) + " |")
    for name, records in [("all_tokens.csv", rows), ("reasoning_units.csv", units)]:
        if records:
            with (args.output_dir / name).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(metadata)} complete trajectories, {len(rows)} tokens, {len(units)} contiguous units.")


if __name__ == "__main__":
    main()
