#!/usr/bin/env python3
"""Compare Teacher visual contexts on saved rollouts, without updating weights."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import torch
import numpy as np
from PIL import Image

from probe_vstar_advantages import append_json, make_prompt, read_json, read_lines, score_inputs, write_json


VARIANTS = ("crop_with_focus", "crop_only")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(args):
    base = read_json(args.source / "manifest.json")
    audit = read_json(args.source / "crop_audit.json")
    config = {
        "source": str(args.source),
        "source_sha256": {name: sha256(args.source / name) for name in
                          ("manifest.json", "rollouts.jsonl", "scores.jsonl", "advantages.pt", "crop_audit.json")},
        "model": base["model"],
        "variants": list(VARIANTS),
        "relation_question_ids": [r["extra_info"]["question_id"] for r in base["selected"]
                                  if r["extra_info"]["category"] == "relative_position"],
        "invalid_crop_question_ids": [str(x) for x in audit["failed_question_ids"]],
        "relation_policy": "Original Student prompt and image; no focus or crop; OPSD disabled.",
        "invalid_crop_policy": "Score as a separate failure control; exclude from primary analysis and applied OPSD.",
        "visual_preprocessing": "Reuse the exact crop pixel tensors and grids from the baseline Teacher prefix.",
        "student_scores": "Reuse saved Transformers log-probabilities; verify a Student and baseline Teacher rescore.",
        "opsd_coef": base["opsd_coef"],
        "optimizer_steps": 0,
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "vllm")},
    }
    assert config["versions"] == read_json(args.source / "versions.json")
    target = args.output / "comparison_manifest.json"
    if target.exists():
        assert read_json(target) == config, "Comparison configuration changed"
    write_json(target, config)
    for variant in VARIANTS:
        (args.output / variant / "inputs").mkdir(parents=True, exist_ok=True)
    print(json.dumps(config), flush=True)


def prefix_from_vision(processor, prompt, pixel_values, image_grid_thw):
    """Use the processor's own expansion with already processed visual patches."""
    parts = prompt.split(processor.image_token)
    assert len(parts) - 1 == len(image_grid_thw)
    expanded = parts[0]
    for i, suffix in enumerate(parts[1:]):
        expanded += processor.replace_image_token({"image_grid_thw": image_grid_thw}, i) + suffix
    prefix = dict(processor.tokenizer([expanded], return_tensors="pt"))
    prefix["mm_token_type_ids"] = torch.tensor(processor.create_mm_token_type_ids(prefix["input_ids"].tolist()))
    prefix.update(pixel_values=pixel_values, image_grid_thw=image_grid_thw)
    return prefix


def score(args):
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
    from groove.evidence import teacher_payload
    from groove.schemas import TeacherEvidence
    from groove.verl_trainer import GrooveRayPPOTrainer

    config = read_json(args.output / "comparison_manifest.json")
    base = read_json(args.source / "manifest.json")
    assert config["source_sha256"]["scores.jsonl"] == sha256(args.source / "scores.jsonl")
    rows = {str(r["extra_info"]["question_id"]): r for r in base["selected"]}
    output = args.output / args.variant
    result_path = output / "scores.jsonl"
    completed = {r["question_id"] for r in read_lines(result_path)} if result_path.exists() else set()
    processor = AutoProcessor.from_pretrained(base["model"], local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        base["model"], local_files_only=True, dtype=torch.bfloat16, attn_implementation="flash_attention_2",
    ).eval().cuda()
    model.requires_grad_(False)
    adapter = object.__new__(GrooveRayPPOTrainer)
    adapter.processor = processor
    checked = False
    for record in read_lines(args.source / "scores.jsonl"):
        qid = record["question_id"]
        if qid in completed:
            continue
        started = time.monotonic()
        row = rows[qid]
        relation = qid in config["relation_question_ids"]
        invalid = qid in config["invalid_crop_question_ids"]
        if relation:
            # Identical weights and identical input give identical distributions.
            # Use their saved scores exactly, without introducing numerical noise.
            teacher_prefix = None
            teacher_prompt = record["prompt"]
            teacher_tokens = record["student_prompt_tokens"]
            teacher_images = [row["extra_info"]["image_path"]]
            grid = None
            mode = "original_only"
        else:
            ev_dir = args.source / "evidence" / f"vstar-{qid}"
            ev = TeacherEvidence.model_validate_json((ev_dir / "evidence.json").read_text())
            template, refs = teacher_payload(ev, question=row["extra_info"]["question"])
            images = [Image.open(r["path"]).convert("RGB") for r in refs]
            messages = adapter._build_teacher_messages_from_template(template, images)
            old_prompt, old_prefix = adapter._process_teacher_multimodal_prompt(
                messages, images, {"enable_thinking": False}, base["teacher_max_prompt_len"])
            saved = read_json(ev_dir / "scored_prompt.json")
            assert old_prefix["input_ids"][0].tolist() == saved["prompt_token_ids"]
            assert old_prefix["image_grid_thw"].tolist() == saved["image_grid_thw"]
            rebuilt = prefix_from_vision(processor, old_prompt, old_prefix["pixel_values"], old_prefix["image_grid_thw"])
            assert rebuilt.keys() == old_prefix.keys()
            assert all(torch.equal(rebuilt[k], old_prefix[k]) for k in rebuilt), "Processor reconstruction differs"
            assert len(ev.crops) == 1, "This diagnostic handles local single-crop questions only"
            patch_counts = old_prefix["image_grid_thw"].prod(-1)
            assert int(patch_counts.sum()) == old_prefix["pixel_values"].shape[0]
            crop_pixels = old_prefix["pixel_values"][int(patch_counts[0]):].clone()
            crop_grid = old_prefix["image_grid_thw"][1:].clone()
            if args.variant == "crop_only":
                # Preserve the Student's question and response instruction exactly.
                template = row["prompt"]
            else:
                # Remove only the original image placeholder from the baseline.
                text = ev.teacher_prompt[0]["content"]
                assert text.startswith("<image>\n")
                template = [{"role": "user", "content": text[len("<image>\n"):]}]
            teacher_images = [str(c.path) for c in ev.crops]
            messages = adapter._build_teacher_messages_from_template(template, images[1:])
            teacher_prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            teacher_prefix = prefix_from_vision(processor, teacher_prompt, crop_pixels, crop_grid)
            teacher_tokens = teacher_prefix["input_ids"].shape[-1]
            assert teacher_tokens <= base["teacher_max_prompt_len"]
            grid = crop_grid.tolist()
            mode = args.variant
            if not checked:
                prompt, image = make_prompt(processor, row)
                student_prefix = dict(processor(text=[prompt], images=[image], return_tensors="pt"))
                assert student_prefix["input_ids"][0].tolist() == record["prompt_token_ids"]
                first = record["completions"][0]
                checks = {}
                for name, prefix in (("student", student_prefix), ("teacher", old_prefix)):
                    fresh = score_inputs(model, prefix, first["token_ids"])
                    delta = (fresh - torch.tensor(first[f"{name}_log_probs"])).abs()
                    checks[name + "_max_abs_delta"] = delta.max().item()
                    assert delta.max().item() < 1e-4, checks
                write_json(output / "rescore_validation.json", {"question_id": qid, **checks})
                checked = True
        write_json(output / "inputs" / f"{qid}.json", {
            "question_id": qid, "mode": mode, "image_paths": teacher_images,
            "prompt": teacher_prompt, "prompt_tokens": teacher_tokens,
            "prompt_token_ids": teacher_prefix["input_ids"][0].tolist() if teacher_prefix else record["prompt_token_ids"],
            "image_grid_thw": grid, "crop_pixels_identical_to_baseline": not relation,
            "primary_analysis": not relation and not invalid, "invalid_crop": invalid,
        })
        completions = []
        for item in record["completions"]:
            teacher = (score_inputs(model, teacher_prefix, item["token_ids"]).tolist()
                       if teacher_prefix else list(item["student_log_probs"]))
            completions.append({**item, "teacher_log_probs": teacher})
        append_json(result_path, {
            **record, "completions": completions, "teacher_prompt_tokens": teacher_tokens,
            "teacher_mode": mode, "primary_analysis": not relation and not invalid,
            "opsd_enabled": not relation and not invalid, "invalid_crop": invalid,
            "evidence_status": "ready" if not relation and not invalid else "skipped",
        })
        print(json.dumps({"variant": args.variant, "question": qid, "mode": mode,
                          "prompt_tokens": teacher_tokens, "seconds": time.monotonic() - started}), flush=True)


def write_csv(path, rows):
    with path.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_signs(values):
    return {"n": len(values), "positive": sum(x > 0 for x in values),
            "negative": sum(x < 0 for x in values), "zero": sum(x == 0 for x in values),
            "positive_above_0.01": sum(x > .01 for x in values),
            "negative_below_minus_0.01": sum(x < -.01 for x in values),
            "near_zero": sum(abs(x) <= .01 for x in values)}


def report(args):
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    from groove.losses import groove_opsd_advantages, combine_grpo_opsd_advantages
    from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage, compute_policy_loss_vanilla

    config = read_json(args.output / "comparison_manifest.json")
    for name, expected in config["source_sha256"].items():
        assert sha256(args.source / name) == expected, f"Source changed: {name}"
    baseline = read_lines(args.source / "scores.jsonl")
    old_tensors = torch.load(args.source / "advantages.pt", weights_only=False, map_location="cpu")
    excluded = set(config["relation_question_ids"] + config["invalid_crop_question_ids"])
    primary_ids = [r["question_id"] for r in baseline if r["question_id"] not in excluded]
    tokenizer = AutoTokenizer.from_pretrained(config["model"], local_files_only=True)
    annotations = read_json(args.source / "final_answer_token_audit.json")
    results, lookups, token_maps, group_rows = {}, {}, {}, []
    for variant in ("baseline", *VARIANTS):
        records = baseline if variant == "baseline" else read_lines(args.output / variant / "scores.jsonl")
        assert [r["question_id"] for r in records] == [r["question_id"] for r in baseline]
        pairs = [(r, c) for r in records for c in r["completions"]]
        originals = [(r, c) for r in baseline for c in r["completions"]]
        shape = old_tensors["student_log_probs"].shape
        student, teacher, mask, rewards = [torch.zeros(shape) for _ in range(4)]
        enabled, group_ids = [], []
        for i, ((record, item), (_, old)) in enumerate(zip(pairs, originals, strict=True)):
            for key in ("token_ids", "student_log_probs", "reward", "completion"):
                assert item[key] == old[key], f"Changed rollout field: {key}"
            n = len(item["token_ids"])
            student[i, :n] = torch.tensor(item["student_log_probs"])
            teacher[i, :n] = torch.tensor(item["teacher_log_probs"])
            mask[i, :n] = 1
            rewards[i, n - 1] = item["reward"]["score"]
            enabled.append(record["question_id"] in primary_ids)
            group_ids.append(record["question_id"])
        enabled = torch.tensor(enabled)
        grpo, _ = compute_grpo_outcome_advantage(rewards, mask, np.array(group_ids))
        assert torch.equal(grpo, old_tensors["grpo_advantages"])
        assert torch.equal(student, old_tensors["student_log_probs"])
        assert torch.equal(mask, old_tensors["response_mask"])
        raw, _ = groove_opsd_advantages(student, teacher, mask)
        applied, _ = groove_opsd_advantages(student, teacher, mask, evidence_mask=enabled)
        total = combine_grpo_opsd_advantages(grpo, applied, opsd_coef=config["opsd_coef"])
        loss_cfg = OmegaConf.create(dict(clip_ratio=.2, clip_ratio_low=.2, clip_ratio_high=.2,
                                        clip_ratio_c=3., global_batch_info={}))
        current = student.clone().requires_grad_(True)
        loss, _ = compute_policy_loss_vanilla(student, current, total, mask, config=loss_cfg)
        loss.backward()
        assert torch.allclose(current.grad, -total / mask.sum(), atol=1e-7)
        if variant != "baseline":
            relation_rows = torch.tensor([q in config["relation_question_ids"] for q in group_ids])
            assert torch.count_nonzero(raw[relation_rows]) == 0
        output = args.output / variant
        output.mkdir(exist_ok=True)
        torch.save({"student_log_probs": student, "teacher_log_probs": teacher, "response_mask": mask,
                    "evidence_mask": enabled, "grpo_advantages": grpo, "raw_opsd_advantages": raw,
                    "applied_opsd_advantages": applied, "total_advantages": total,
                    "logprob_gradient": current.grad, "group_ids": group_ids}, output / "advantages.pt")
        token_rows = []
        lookup = {(r["question_id"], c["rollout_id"]): c for r, c in pairs}
        for i, (record, item) in enumerate(pairs):
            for j, token_id in enumerate(item["token_ids"]):
                token_rows.append({"question_id": record["question_id"], "rollout_id": item["rollout_id"],
                    "position": j, "token_id": token_id, "token": tokenizer.decode([token_id]),
                    "primary_analysis": bool(enabled[i]), "student_log_prob": student[i, j].item(),
                    "teacher_log_prob": teacher[i, j].item(), "grpo": grpo[i, j].item(),
                    "raw_opsd": raw[i, j].item(), "applied_opsd": applied[i, j].item(),
                    "weighted_opsd": config["opsd_coef"] * applied[i, j].item(), "total": total[i, j].item()})
        write_csv(output / "tokens.csv", token_rows)
        token_maps[variant] = {(r["question_id"], r["rollout_id"], r["position"]): r for r in token_rows}
        primary = (mask.bool() & enabled[:, None])
        values = raw[primary]
        metrics = {"primary_question_ids": primary_ids, "primary_rollouts": int(enabled.sum()),
                   "primary_tokens": int(primary.sum()), "raw_opsd_mean": values.mean().item(),
                   "raw_opsd_rms": values.square().mean().sqrt().item(),
                   "weighted_opsd_rms": config["opsd_coef"] * values.square().mean().sqrt().item(),
                   "grpo_rms": grpo[primary].square().mean().sqrt().item(),
                   "policy_loss_with_primary_opsd_only": loss.item(), "optimizer_steps": 0}
        for correct in (True, False):
            selected = [r for r in annotations if r["question_id"] in primary_ids and r["correct"] == correct]
            gaps = [token_maps[variant][(r["question_id"], int(r["rollout_id"]), int(r["position"]))]["raw_opsd"]
                    for r in selected]
            metrics["correct_final_tokens" if correct else "wrong_final_tokens"] = summarize_signs(gaps)
        for qid in primary_ids:
            subset = [r for r in token_rows if r["question_id"] == qid]
            arr = np.array([r["raw_opsd"] for r in subset])
            group_rows.append({"question_id": qid, "variant": variant, "tokens": len(subset),
                               "opsd_mean": float(arr.mean()), "opsd_rms": float(np.sqrt((arr**2).mean()))})
        write_json(output / "metrics.json", metrics)
        results[variant] = metrics
        lookups[variant] = lookup
    examples = []
    for row in csv.DictReader((args.source / "token_credit_examples.csv").open()):
        key = (row["question_id"], int(row["rollout_id"]), int(row["position"]))
        entry = {"question_id": key[0], "rollout_id": key[1], "position": key[2], "token": row["token"],
                 "interpretation": row["interpretation"], "primary_analysis": key[0] in primary_ids,
                 "student_log_prob": float(row["student_log_prob"]), "grpo": float(row["grpo"])}
        for variant in results:
            token = token_maps[variant][key]
            assert token["token_id"] == int(row["token_id"])
            entry[variant + "_opsd"] = token["raw_opsd"]
            entry[variant + "_teacher_probability"] = float(np.exp(token["teacher_log_prob"]))
            entry[variant + "_total"] = token["total"]
        examples.append(entry)
    final_rows = []
    for row in annotations:
        key = (row["question_id"], int(row["rollout_id"]), int(row["position"]))
        entry = {"question_id": key[0], "rollout_id": key[1], "position": key[2], "token": row["token"],
                 "correct": row["correct"], "primary_analysis": key[0] in primary_ids}
        for variant in results:
            entry[variant + "_opsd"] = token_maps[variant][key]["raw_opsd"]
        final_rows.append(entry)
    write_csv(args.output / "token_credit_comparison.csv", examples)
    write_csv(args.output / "final_answer_comparison.csv", final_rows)
    write_csv(args.output / "groups.csv", group_rows)
    write_json(args.output / "comparison_metrics.json", results)
    write_json(args.output / "validation.json", {
        "source_hashes_unchanged": True, "rollouts_rewards_and_student_scores_unchanged": True,
        "grpo_exactly_unchanged": True, "baseline_processor_reconstruction_exact": True,
        "crop_pixel_tensors_and_grids_reused": True, "relation_raw_opsd_exactly_zero": True,
        "production_advantage_and_policy_loss_used": True, "logprob_gradient_check_passed": True,
        "optimizer_steps": 0, "questions": len(baseline), "rollouts": len(originals),
        "variants": {v: read_json(args.output / v / "rescore_validation.json") for v in VARIANTS},
    })
    labels = {"baseline": "原图 + Crop + focus", "crop_with_focus": "Crop + focus", "crop_only": "仅 Crop + 原题"}
    lines = ["# VStar Teacher 图像输入对照", "",
             "结果：仅 Crop + 原题改善了这批错误最终选项的信号，也修正了部分关键内容词；但正确词受罚和错误词获正信号仍存在。保留 focus、只删原图的对照没有表现出一致改善。", "",
             "复用同一批 10 题 / 80 条 Student rollout。Student 与 Teacher 均为同一份更新前 Qwen3.5-2B；不重新采样、不更新模型，不切换 EMA 或分布蒸馏目标。", "",
             "主分析为 7 道局部题 / 56 条轨迹。Q128、Q125 关系题仅保留原图和原始 Student 提示，禁用 OPSD；Q101 错误 Crop 单独打分留档，排除主分析并禁用其 OPSD。", "",
             "对照一：仅移除 Teacher 原图，保留 focus 和其他文字。对照二：仅使用 Crop，并完全沿用 Student 的原题及回答要求，不添加 focus。所有 Crop 使用与旧 Teacher 完全相同的预处理视觉张量和网格；两组均沿用同一条 Student 生成前缀。", "",
             "A_OPSD = log p_T(已采样 token) − log p_S(已采样 token)，表内未乘 0.01。GRPO 完全不变。正负是概率变化方向，不等价于视觉判断对错，也不是训练后准确率。", "",
             "## 最终选项 token（主分析）", "",
             "47 条选对、7 条明确选错；另 2 条有正确颜色描述但输出不符合选项解析要求，未混入明确选错统计。", "",
             "| Teacher 输入 | 正确选项：正 / 负 | 错误选项：正 / 负 | 原始 OPSD RMS |", "|---|---:|---:|---:|"]
    for variant, metrics in results.items():
        good, bad = metrics["correct_final_tokens"], metrics["wrong_final_tokens"]
        lines.append(f"| {labels[variant]} | {good['positive']} / {good['negative']} | {bad['positive']} / {bad['negative']} | {metrics['raw_opsd_rms']:.6f} |")
    lines += ["", "按 |A_OPSD| > 0.01 过滤微小变化：", "",
              "| Teacher 输入 | 正确选项：明显正 / 明显负 / 接近零 | 错误选项：明显正 / 明显负 / 接近零 |",
              "|---|---:|---:|"]
    for variant, metrics in results.items():
        good, bad = metrics["correct_final_tokens"], metrics["wrong_final_tokens"]
        fmt = lambda x: f"{x['positive_above_0.01']} / {x['negative_below_minus_0.01']} / {x['near_zero']}"
        lines.append(f"| {labels[variant]} | {fmt(good)} | {fmt(bad)} |")
    lines += ["", "## 同一个关键 token 的比较", "",
              "所有 R 和 position 从 0 开始。以下案例沿用此前已挑选的案例，没有依据新结果重新选择词例。", "",
              "| 题目 / R / position | token | 原图 + Crop + focus | Crop + focus | 仅 Crop + 原题 |",
              "|---|---|---:|---:|---:|"]
    for row in examples:
        if not row["primary_analysis"]:
            continue
        values = " | ".join(f"{row[v + '_opsd']:+.6f}" for v in results)
        lines.append(f"| {row['question_id']} / {row['rollout_id']} / {row['position']} | `{row['token']}` | {values} |")
    lines += ["", "## 案例上下文", ""]
    seen = set()
    for row in examples:
        key = (row["question_id"], row["rollout_id"])
        if key in seen or not row["primary_analysis"]:
            continue
        seen.add(key)
        item = lookups["baseline"][key]
        lines += [f"### Q{key[0]} / R{key[1]}", "", "旧设置下的观察：" + row["interpretation"], "",
                  *["> " + line for line in item["completion"].splitlines()], ""]
    lines += ["## 验证与边界", "",
              "源文件哈希、全部生成 token、reward、Student log-prob、GRPO 均保持不变。重算一条 Student 和旧 Teacher 分数核对模型与数值环境；逐题复原旧 Teacher 输入 token，且新 Crop 分支直接复用其视觉张量。使用生产 advantage/PPO loss 函数，并验证 loss 对 log-prob 的梯度。", "",
              "本次检验的是 Teacher 输入方式。没有重新生成 Teacher 答案，也没有执行模型参数 backward 或 optimizer.step，因此不能据此声称训练准确率提高。只有 log-prob 叶子张量进行了梯度校验。", "",
              "7 个错误最终选项均为负信号，仍只是这 7 个位置的结果；其中部分位置的 Student 前缀已经包含错误结论。正确内容词仍可能得到负信号，错误内容词也仍可能得到正信号。", "",
              f"[逐 token 比较]({args.output / 'token_credit_comparison.csv'})；[最终选项全量比较]({args.output / 'final_answer_comparison.csv'})；[指标]({args.output / 'comparison_metrics.json'})；[验证]({args.output / 'validation.json'})。", ""]
    (args.output / "report.md").write_text("\n".join(lines))
    print(json.dumps(results), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("prepare", "score", "report"), required=True)
    parser.add_argument("--variant", choices=VARIANTS)
    args = parser.parse_args()
    args.source = args.source.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage == "score" and args.variant is None:
        parser.error("score requires --variant")
    globals()[args.stage](args)


if __name__ == "__main__":
    main()
