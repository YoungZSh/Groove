#!/usr/bin/env python3
"""Audit real VStar rollouts, visual evidence and pre-update GRPO/OPSD targets.

Run stages in order: select, rollout, evidence, score, report. Model weights are
fixed for this diagnostic batch; no optimizer step or checkpoint is written.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from groove.reward import compute_score


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def read_json(path):
    return json.loads(path.read_text())


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append_json(path, value):
    with path.open("a") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def select(args):
    import pyarrow.parquet as pq

    rows = pq.read_table(args.data).to_pylist()
    indices = random.Random(args.seed).sample(range(len(rows)), args.questions)
    selected = []
    for index in indices:
        row = rows[index]
        image_path = Path(row["extra_info"]["image_path"])
        selected.append({"row_index": index, "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(), **row})
    manifest = {
        "seed": args.seed, "dataset": str(args.data.resolve()), "dataset_rows": len(rows),
        "model": str(args.model.resolve()), "n": args.n, "max_new_tokens": args.max_new_tokens,
        "temperature": 1.0, "top_p": 1.0, "top_k": -1, "enable_thinking": False,
        "opsd_coef": args.opsd_coef, "opsd_clip": None, "teacher_max_prompt_len": 9216,
        "teacher_weights": "same pre-update actor; no EMA", "optimizer_steps": 0,
        "selected": selected,
    }
    path = args.output / "manifest.json"
    if path.exists() and read_json(path) != manifest:
        raise ValueError("Existing manifest differs; choose a new output directory")
    write_json(path, manifest)
    print(json.dumps({"indices": indices, "questions": len(selected), "rollouts": len(selected) * args.n}), flush=True)


def make_prompt(processor, row):
    from groove.verl_trainer import GrooveRayPPOTrainer

    # Reuse production placeholder handling, without starting a Ray trainer.
    adapter = object.__new__(GrooveRayPPOTrainer)
    adapter.processor = processor
    image = Image.open(row["extra_info"]["image_path"]).convert("RGB")
    messages = adapter._build_teacher_messages_from_template(row["prompt"], [image])
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return prompt, image


def rollout(args, manifest):
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    output_path = args.output / "rollouts.jsonl"
    completed = {x["row_index"] for x in read_lines(output_path)} if output_path.exists() else set()
    pending = [row for row in manifest["selected"] if row["row_index"] not in completed]
    if not pending:
        return
    processor = AutoProcessor.from_pretrained(manifest["model"], local_files_only=True)
    llm = LLM(
        model=manifest["model"], tensor_parallel_size=1, dtype="bfloat16",
        max_model_len=32768, max_num_seqs=16, max_num_batched_tokens=8192,
        gpu_memory_utilization=0.55, enforce_eager=True, skip_mm_profiling=True,
        limit_mm_per_prompt={"image": 1}, seed=manifest["seed"],
    )
    for row in pending:
        prompt, image = make_prompt(processor, row)
        seed = manifest["seed"] + row["row_index"]
        result = llm.generate(
            [{"prompt": prompt, "multi_modal_data": {"image": image}}],
            SamplingParams(n=manifest["n"], temperature=manifest["temperature"],
                           top_p=manifest["top_p"], top_k=manifest["top_k"],
                           max_tokens=manifest["max_new_tokens"], logprobs=1, seed=seed),
            use_tqdm=False,
        )[0]
        completions = []
        for index, item in enumerate(result.outputs):
            reward = compute_score("vstar_groove", item.text, row["extra_info"]["answer"])
            completions.append({
                "rollout_id": index, "completion": item.text, "token_ids": list(item.token_ids),
                "rollout_log_probs": [float(lp[token].logprob) for token, lp in zip(item.token_ids, item.logprobs, strict=True)],
                "finish_reason": item.finish_reason, "reward": reward,
            })
        record = {"row_index": row["row_index"], "question_id": row["extra_info"]["question_id"],
                  "sampling_seed": seed, "prompt": prompt, "prompt_token_ids": list(result.prompt_token_ids),
                  "completions": completions}
        append_json(output_path, record)
        print(json.dumps({"stage": "rollout", "question": record["question_id"], "prompt_tokens": len(result.prompt_token_ids),
                          "rewards": [x["reward"]["score"] for x in completions],
                          "lengths": [len(x["token_ids"]) for x in completions]}), flush=True)


def evidence(args, manifest):
    from groove.analyzer import OpenAIAnalyzerConfig, OpenAICompatibleAnalyzer
    from groove.evidence import EvidenceBuilderConfig, TeacherEvidenceBuilder
    from groove.analyzer_tools import AnalyzerVisionToolRegistry
    from groove.grounding import crop_tool_regions
    from groove.schemas import GroupRollout, Rollout, ToolRegion

    rows = {row["row_index"]: row for row in manifest["selected"]}
    analyzer = OpenAICompatibleAnalyzer(OpenAIAnalyzerConfig.from_env())
    if not os.environ.get("ANALYZER_GROUNDING_URL") or not os.environ.get("ANALYZER_OCR_URL"):
        raise ValueError("This probe requires the configured remote DINO and OCR services")
    registry = AnalyzerVisionToolRegistry()

    class RemoteGrounder:
        def crop_objects(self, image_path, focus, output_dir):
            regions, trace = [], []
            for query in focus.grounding_queries:
                result = registry.ground_image(image_path, query, focus.context_margin)
                trace.append({"query": query, "result": result})
                if result.get("found"):
                    regions.append(ToolRegion(query=query, expanded_box=result["bbox"],
                                              score=result["score"], source="ground_image"))
            output_dir.mkdir(parents=True, exist_ok=True)
            write_json(output_dir / "fallback_grounding_trace.json", trace)
            return crop_tool_regions(image_path, regions, output_dir)

    grounder = RemoteGrounder()
    write_json(args.output / "services.json", {
        "analyzer_model": analyzer.config.model, "analyzer_base_url": analyzer.config.base_url,
        "grounding_url": os.environ["ANALYZER_GROUNDING_URL"], "ocr_url": os.environ["ANALYZER_OCR_URL"],
        "use_vision_tools": analyzer.config.use_vision_tools, "max_tool_rounds": analyzer.config.max_tool_rounds,
        "max_completion_tokens": analyzer.config.max_completion_tokens,
    })
    builder = TeacherEvidenceBuilder(analyzer, grounder, EvidenceBuilderConfig(args.output / "evidence"))
    for record in read_lines(args.output / "rollouts.jsonl"):
        row = rows[record["row_index"]]
        group = GroupRollout(uid=f"vstar-{record['question_id']}", question=row["extra_info"]["question"],
                             image_path=Path(row["extra_info"]["image_path"]),
                             rollouts=[Rollout(rollout_id=x["rollout_id"], completion=x["completion"],
                                               predicted_label=x["reward"]["predicted_label"], reward=x["reward"]["score"])
                                       for x in record["completions"]])
        analyzer.last_tool_trace = []
        started = time.monotonic()
        result = builder.build(group)
        audit_path = args.output / "evidence" / group.uid / "tool_trace.json"
        if analyzer.last_tool_trace or not audit_path.exists():
            write_json(audit_path, analyzer.last_tool_trace)
        print(json.dumps({"stage": "evidence", "question": record["question_id"], "status": result.status,
                          "crops": len(result.crops), "reason": result.reason, "seconds": time.monotonic() - started}), flush=True)


def score_inputs(model, prefix, response_ids):
    length = len(response_ids)
    response = torch.tensor([response_ids], device=model.device)
    inputs = {key: value.to(model.device) for key, value in prefix.items()}
    inputs["input_ids"] = torch.cat((inputs["input_ids"], response), dim=-1)
    inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
    if "mm_token_type_ids" in inputs:
        inputs["mm_token_type_ids"] = torch.cat((inputs["mm_token_type_ids"], torch.zeros_like(response)), dim=-1)
    with torch.inference_mode():
        # Only materialize response logits; long image prefixes have a large vocabulary.
        logits = model(**inputs, use_cache=False, logits_to_keep=length + 1).logits[:, :-1].float()
        scores = logits.log_softmax(-1).gather(-1, response.unsqueeze(-1)).squeeze(-1)[0].cpu()
    if scores.shape != (length,) or not scores.isfinite().all():
        raise ValueError("Invalid teacher-forced token scores")
    return scores


def score(args, manifest):
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
    from groove.evidence import teacher_payload
    from groove.schemas import TeacherEvidence
    from groove.verl_trainer import GrooveRayPPOTrainer

    output_path = args.output / "scores.jsonl"
    completed = {x["question_id"] for x in read_lines(output_path)} if output_path.exists() else set()
    processor = AutoProcessor.from_pretrained(manifest["model"], local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        manifest["model"], local_files_only=True, dtype=torch.bfloat16, attn_implementation="flash_attention_2",
    ).eval().cuda()
    model.requires_grad_(False)
    adapter = object.__new__(GrooveRayPPOTrainer)
    adapter.processor = processor
    rows = {row["row_index"]: row for row in manifest["selected"]}
    for record in read_lines(args.output / "rollouts.jsonl"):
        if record["question_id"] in completed:
            continue
        started = time.monotonic()
        row = rows[record["row_index"]]
        prompt, image = make_prompt(processor, row)
        student_prefix = dict(processor(text=[prompt], images=[image], return_tensors="pt"))
        assert prompt == record["prompt"]
        assert student_prefix["input_ids"][0].tolist() == record["prompt_token_ids"], "Rollout/scoring prompt mismatch"
        evidence_path = args.output / "evidence" / f"vstar-{record['question_id']}" / "evidence.json"
        ev = TeacherEvidence.model_validate_json(evidence_path.read_text())
        teacher_prefix = None
        if ev.status == "ready":
            messages, image_refs = teacher_payload(ev, question=row["extra_info"]["question"])
            images = [Image.open(x["path"]).convert("RGB") for x in image_refs]
            messages = adapter._build_teacher_messages_from_template(messages, images)
            teacher_prompt, teacher_prefix = adapter._process_teacher_multimodal_prompt(
                messages, images, {"enable_thinking": False}, manifest["teacher_max_prompt_len"])
            write_json(evidence_path.parent / "scored_prompt.json", {
                "prompt": teacher_prompt, "prompt_token_ids": teacher_prefix["input_ids"][0].tolist(),
                "image_grid_thw": teacher_prefix["image_grid_thw"].tolist(),
            })
        completions = []
        for item in record["completions"]:
            student = score_inputs(model, student_prefix, item["token_ids"])
            teacher = score_inputs(model, teacher_prefix, item["token_ids"]) if teacher_prefix else student.clone()
            completions.append({**item, "student_log_probs": student.tolist(), "teacher_log_probs": teacher.tolist()})
        append_json(output_path, {**record, "completions": completions, "evidence_status": ev.status,
                                 "student_prompt_tokens": student_prefix["input_ids"].shape[-1],
                                 "teacher_prompt_tokens": teacher_prefix["input_ids"].shape[-1] if teacher_prefix else 0})
        print(json.dumps({"stage": "score", "question": record["question_id"], "seconds": time.monotonic() - started,
                          "evidence": ev.status}), flush=True)


def report(args, manifest):
    from omegaconf import OmegaConf
    from groove.losses import groove_opsd_advantages, combine_grpo_opsd_advantages
    from groove.advantage_metrics import compute_advantage_metrics
    from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage, compute_policy_loss_vanilla

    records = read_lines(args.output / "scores.jsonl")
    expected_ids = {row["extra_info"]["question_id"] for row in manifest["selected"]}
    assert {r["question_id"] for r in records} == expected_ids
    assert all(len(r["completions"]) == manifest["n"] for r in records)
    flat = [(r, item) for r in records for item in r["completions"]]
    shape = (len(flat), max(len(item["token_ids"]) for _, item in flat))
    student, teacher, mask, rewards, ready = [torch.zeros(shape) for _ in range(5)]
    group_ids = []
    for i, (record, item) in enumerate(flat):
        n = len(item["token_ids"])
        student[i, :n] = torch.tensor(item["student_log_probs"])
        teacher[i, :n] = torch.tensor(item["teacher_log_probs"])
        mask[i, :n] = 1
        rewards[i, n - 1] = item["reward"]["score"]
        ready[i, :n] = float(record["evidence_status"] == "ready")
        group_ids.append(record["question_id"])
    grpo, _ = compute_grpo_outcome_advantage(rewards, mask, np.array(group_ids))
    opsd, _ = groove_opsd_advantages(student, teacher, mask, evidence_mask=ready.any(-1))
    total = combine_grpo_opsd_advantages(grpo, opsd, opsd_coef=manifest["opsd_coef"])
    metrics = compute_advantage_metrics(
        grpo_advantages=grpo, opsd_advantages=opsd, total_advantages=total,
        student_log_probs=student, teacher_log_probs=teacher, response_mask=mask, evidence_mask=ready.any(-1),
        opsd_coef=manifest["opsd_coef"], sequence_rewards=rewards.sum(-1), group_ids=group_ids)
    config = OmegaConf.create(dict(clip_ratio=.2, clip_ratio_low=.2, clip_ratio_high=.2, clip_ratio_c=3., global_batch_info={}))
    current = student.detach().clone().requires_grad_(True)
    loss, loss_metrics = compute_policy_loss_vanilla(student, current, total, mask, config=config)
    loss.backward()
    assert torch.allclose(current.grad, -total / mask.sum(), atol=1e-7)
    metrics.update(loss_metrics)
    metrics.update({"diagnostic/initial_policy_loss": loss.item(),
                    "diagnostic/initial_grpo_loss": -grpo[mask.bool()].mean().item(),
                    "diagnostic/initial_weighted_opsd_loss": -(manifest["opsd_coef"] * opsd[mask.bool()]).mean().item(),
                    "diagnostic/logprob_gradient_rms": current.grad[mask.bool()].square().mean().sqrt().item(),
                    "diagnostic/optimizer_steps": 0, "diagnostic/theoretical_reference_kl_at_identical_weights": 0.0})
    write_json(args.output / "metrics.json", metrics)
    torch.save(dict(student_log_probs=student, teacher_log_probs=teacher, response_mask=mask, evidence_mask=ready,
                    grpo_advantages=grpo, opsd_advantages=opsd, total_advantages=total,
                    logprob_gradient=current.grad, sequence_rewards=rewards.sum(-1), group_ids=group_ids), args.output / "advantages.pt")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(manifest["model"], local_files_only=True)
    trajectory_rows, token_rows = [], []
    for i, (record, item) in enumerate(flat):
        n = len(item["token_ids"])
        raw = opsd[i, :n]
        discrepancy = student[i, :n] - torch.tensor(item["rollout_log_probs"])
        trajectory_rows.append({"question_id": record["question_id"], "rollout_id": item["rollout_id"],
            "answer": item["reward"]["predicted_label"], "correct": item["reward"]["accuracy"], "reward": item["reward"]["score"],
            "token_count": n, "evidence_status": record["evidence_status"], "grpo": grpo[i, 0].item(),
            "opsd_mean": raw.mean().item(), "opsd_rms": raw.square().mean().sqrt().item(),
            "opsd_min": raw.min().item(), "opsd_max": raw.max().item(),
            "weighted_opsd_mean": (manifest["opsd_coef"] * raw).mean().item(),
            "total_mean": total[i, :n].mean().item(), "rollout_rescore_mae": discrepancy.abs().mean().item()})
        for j, token in enumerate(item["token_ids"]):
            token_rows.append({"question_id": record["question_id"], "rollout_id": item["rollout_id"], "position": j,
                "token_id": token, "token": tokenizer.decode([token]), "reward": item["reward"]["score"],
                "student_log_prob": student[i, j].item(), "teacher_log_prob": teacher[i, j].item(),
                "grpo": grpo[i, j].item(), "opsd": opsd[i, j].item(),
                "weighted_opsd": (manifest["opsd_coef"] * opsd[i, j]).item(), "total": total[i, j].item()})
    for name, rows in [("trajectories.csv", trajectory_rows), ("tokens.csv", token_rows)]:
        with (args.output / name).open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write_report(args.output, manifest, records, trajectory_rows, token_rows, metrics)
    print(json.dumps({"stage": "report", "questions": len(records), "rollouts": len(flat), "tokens": len(token_rows), "metrics": metrics}), flush=True)


def write_report(output, manifest, records, trajectories, tokens, metrics):
    import importlib.metadata

    versions = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "vllm")}
    write_json(output / "versions.json", versions)
    completions = [item for record in records for item in record["completions"]]
    discrepancy = np.concatenate([np.array(item["student_log_probs"]) - np.array(item["rollout_log_probs"]) for item in completions])
    gaps = np.array([item["opsd"] for item in tokens])
    validation = {
        "questions": len(records), "rollouts": len(completions), "valid_tokens": len(tokens),
        "correct": int(sum(item["reward"]["accuracy"] for item in completions)),
        "format_correct": int(sum(item["reward"]["format_reward"] for item in completions)),
        "truncated": sum(item["finish_reason"] == "length" for item in completions),
        "evidence_ready": sum(record["evidence_status"] == "ready" for record in records),
        "rescore_mae": float(np.abs(discrepancy).mean()), "rescore_abs_p99": float(np.quantile(np.abs(discrepancy), .99)),
        "rescore_abs_max": float(np.abs(discrepancy).max()),
        "opsd_positive_fraction": float((gaps > 0).mean()), "opsd_negative_fraction": float((gaps < 0).mean()),
        "opsd_min": float(gaps.min()), "opsd_max": float(gaps.max()),
    }
    write_json(output / "validation.json", validation)
    groups = []
    for record in records:
        uid = record["question_id"]
        items = [x for x in trajectories if x["question_id"] == uid]
        token_items = [x for x in tokens if x["question_id"] == uid]
        values = np.array([x["opsd"] for x in token_items])
        grpo_values = sorted({round(x["grpo"], 6) for x in items})
        groups.append({"question_id": uid, "correct": int(sum(x["correct"] for x in items)),
                       "n": len(items), "grpo_values": ", ".join(f"{x:+.6f}" for x in grpo_values),
                       "opsd_mean": float(values.mean()), "opsd_rms": float(np.sqrt(np.mean(values ** 2))),
                       "weighted_opsd_rms": manifest["opsd_coef"] * float(np.sqrt(np.mean(values ** 2))),
                       "opsd_min": float(values.min()), "opsd_max": float(values.max()),
                       "evidence_status": record["evidence_status"]})
    with (output / "groups.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(groups[0])); writer.writeheader(); writer.writerows(groups)
    lines = [
        "# VStar：10 题真实 GRPO + OPSD advantage 检查", "",
        f"从 {manifest['dataset_rows']} 题不放回随机抽取 10 题，种子 {manifest['seed']}；每题 {manifest['n']} 条真实采样。",
        "Student / Teacher：同一份 Qwen3.5-2B 更新前权重；Analyzer：远程 Qwen3.8-27B + DINO / OCR HTTP 工具。",
        "Student 只看原图和题目；Teacher 看原图、answer-neutral focus 和独立裁剪。正确答案只用于 reward。",
        f"温度 1、top_p=1、不截断词表采样、关闭 thinking、最多 {manifest['max_new_tokens']} response tokens。",
        "所有有效 response token（含 EOS）参与统计；padding 排除。Student / Teacher log-prob 均用同一 Transformers 模型重新打分。",
        "本轮是更新前诊断：运行采样、评分、证据、双上下文打分、advantage、PPO loss 和对 log-prob 的梯度检查；没有执行模型 backward / optimizer.step。", "",
        "```text", "reward = 0.9 × answer_correct + 0.1 × FINAL_format",
        "A_GRPO = (reward - group_mean) / (group_sample_std + 1e-6)",
        "A_OPSD[t] = stop_gradient(log p_teacher(y_t) - log p_student_old(y_t))",
        "A_total[t] = A_GRPO + 0.01 × evidence_mask × A_OPSD[t]", "```", "",
        "GRPO 使用样本标准差（分母 n−1）；同 reward 组置零。全答对也可能因为 FINAL 格式奖励不同而产生非零 GRPO。表内正确数按现有规则评分器统计，没有人工改判。OPSD 不中心化、不做 trajectory 归一化、无 gap clipping。",
        "这里是采样 token 的有符号 log-prob 差，并非该位置全词表 KL 的精确值。RMS 用于比较尺度，不是额外损失或权重。", "",
        "| 题号 | 正确/8 | GRPO 出现的值 | OPSD 均值 | OPSD RMS | ×0.01 后 RMS | 证据 |",
        "|---|---:|---|---:|---:|---:|---|",
    ]
    for group in groups:
        lines.append(f"| {group['question_id']} | {group['correct']}/{group['n']} | {group['grpo_values']} | {group['opsd_mean']:+.6f} | {group['opsd_rms']:.6f} | {group['weighted_opsd_rms']:.6f} | {group['evidence_status']} |")
    lines += ["", f"全 batch：GRPO RMS = {metrics['grpo/advantage_rms']:.6f}，原始 OPSD RMS = {metrics['opsd/advantage_rms_raw']:.6f}，加权 OPSD RMS = {metrics['opsd/advantage_rms_weighted']:.6f}。",
              f"共 {validation['valid_tokens']} 个有效 token，{validation['correct']}/{validation['rollouts']} 条判对，{validation['format_correct']} 条符合 FINAL 格式，{validation['truncated']} 条截断。OPSD 原始范围 [{validation['opsd_min']:.6f}, {validation['opsd_max']:.6f}]，正值占 {validation['opsd_positive_fraction']:.2%}，负值占 {validation['opsd_negative_fraction']:.2%}。",
              f"加权 OPSD / GRPO 的 token RMS 比例是 {metrics['opsd_to_grpo_advantage_rms_ratio']:.2%}；这不是模型参数梯度比例，也不能据此判断训练后的准确率变化。",
              f"初始 PPO policy loss = {metrics['diagnostic/initial_policy_loss']:.6f}，ratio = 1、clip fraction = 0；其中 GRPO 部分为 {metrics['diagnostic/initial_grpo_loss']:.6f}，加权 OPSD 部分为 {metrics['diagnostic/initial_weighted_opsd_loss']:.6f}。",
              "reference 与初始 actor 权重相同时 reference KL 理论为 0；本诊断未另跑 reference 模型。",
              "loss 的值本身不等于更新强度。已检查其对有效 token log-prob 的梯度等于 −A_total / 有效 token 总数。", "",
              f"vLLM 采样分数与 Transformers 重新打分的绝对差：均值 {validation['rescore_mae']:.6f}、P99 {validation['rescore_abs_p99']:.6f}、最大 {validation['rescore_abs_max']:.6f}。提示 token 已检查完全一致；OPSD 的两分支均采用 Transformers 分数，未将 vLLM Student 分数与 Transformers Teacher 分数混减。", "",
              "详细数据：[80 条轨迹汇总](trajectories.csv)、[全部逐 token 数值](tokens.csv)、[原始生成与双分支打分](scores.jsonl)、[完整张量](advantages.pt)、[指标](metrics.json)。", ""]
    row_map = {row["extra_info"]["question_id"]: row for row in manifest["selected"]}
    for record in records:
        uid = record["question_id"]
        row = row_map[uid]
        ev = read_json(output / "evidence" / f"vstar-{uid}" / "evidence.json")
        lines += [f"## 题号 {uid}", "", row["extra_info"]["question"].replace("\n", "  \n"), "",
                  f"标准答案：{row['extra_info']['answer']}。证据状态：{ev['status']}。", ""]
        if ev.get("focus"):
            lines += [f"Teacher focus：{ev['focus']['visible_focus_instruction']}", ""]
        lines += [f"[原图]({row['extra_info']['image_path']})；" + "；".join(
            f"[裁剪 {i + 1}]({crop['path']})" for i, crop in enumerate(ev.get("crops", []))), ""]
        lines += ["| rollout（从 0 开始） | 回答 | reward | token 数 | GRPO | OPSD 均值 | OPSD RMS | total 均值 |",
                  "|---|---|---:|---:|---:|---:|---:|---:|"]
        for item in [x for x in trajectories if x["question_id"] == uid]:
            lines.append(f"| {item['rollout_id']} | {item['answer']} | {item['reward']:.1f} | {item['token_count']} | {item['grpo']:+.6f} | {item['opsd_mean']:+.6f} | {item['opsd_rms']:.6f} | {item['total_mean']:+.6f} |")
        lines += ["", "第一条 rollout 的尾部 token（包括最终答案，保留实际生成顺序）：", "",
                  "| token | student log p | teacher log p | GRPO | OPSD | 0.01 × OPSD | total |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for item in [x for x in tokens if x["question_id"] == uid and x["rollout_id"] == 0][-10:]:
            token = json.dumps(item["token"], ensure_ascii=False).replace("|", "\\|")
            lines.append(f"| `{token}` | {item['student_log_prob']:.6f} | {item['teacher_log_prob']:.6f} | {item['grpo']:+.6f} | {item['opsd']:+.6f} | {item['weighted_opsd']:+.6f} | {item['total']:+.6f} |")
        lines += ["", "第一条回答：", "", record["completions"][0]["completion"], ""]
    (output / "report.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["select", "rollout", "evidence", "score", "report"], required=True)
    parser.add_argument("--data", type=Path, default=Path("data/vstar/train.parquet"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--questions", type=int, default=10)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--opsd-coef", type=float, default=.01)
    args = parser.parse_args()
    args.output = args.output.resolve(); args.output.mkdir(parents=True, exist_ok=True)
    if args.stage == "select":
        select(args)
    else:
        globals()[args.stage](args, read_json(args.output / "manifest.json"))


if __name__ == "__main__":
    main()
