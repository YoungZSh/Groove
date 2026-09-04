"""Persist complete, aligned rollout evidence and token credit before an update."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch


def write_trajectory_audit(
    output_dir,
    *,
    step,
    batch,
    tokenizer,
    student_log_probs,
    teacher_log_probs,
    grpo_advantages,
    opsd_advantages,
    total_advantages,
    evidence_mask,
    sequence_rewards,
    opsd_coef,
    advantage_clip=None,
):
    """Save every valid response token, including weak credit and EOS.

    Sample IDs refer to this trainer batch after any load-balancing reorder.
    The adjacent JSONL embeds complete responses and source metadata, so it
    does not depend on the ordering of asynchronous rollout log files.
    """
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    mask = batch.batch["response_mask"].detach().cpu().bool()
    responses = batch.batch["responses"].detach().cpu()
    sample_ids, positions = mask.nonzero(as_tuple=True)
    count = mask.shape[0]

    def flatten(value):
        value = value.detach().cpu()
        if value.shape != mask.shape:
            raise ValueError(f"Audit token shape {value.shape} does not match {mask.shape}")
        return value[mask].clone()

    evidence = evidence_mask.detach().cpu().bool()
    rewards = sequence_rewards.detach().cpu()
    payload = {
        "schema_version": 1,
        "step": int(step),
        "scoring_phase": "pre_update_actor",
        "opsd_advantage_coef": float(opsd_coef),
        "opsd_advantage_clip": advantage_clip,
        "sample_count": count,
        "sample_ids": sample_ids,
        "response_positions": positions,
        "token_ids": flatten(responses),
        "student_log_probs": flatten(student_log_probs),
        "teacher_log_probs": flatten(teacher_log_probs),
        "teacher_gaps": flatten(teacher_log_probs - student_log_probs),
        "grpo_advantages": flatten(grpo_advantages),
        "advantages": flatten(grpo_advantages),
        "opsd_advantages": flatten(opsd_advantages),
        "weighted_opsd_advantages": flatten(opsd_coef * opsd_advantages),
        "total_advantages": flatten(total_advantages),
        "evidence_mask": evidence[sample_ids],
        "outcomes": rewards[sample_ids],
    }
    for source_key, target_key in (
        ("rollout_log_probs", "rollout_log_probs"),
        ("ref_log_prob", "ref_log_probs"),
    ):
        if source_key in batch.batch:
            payload[target_key] = flatten(batch.batch[source_key])

    def metadata(name, index, default=None):
        values = batch.non_tensor_batch.get(name)
        return default if values is None else values[index]

    records = []
    for index in range(count):
        extra = metadata("extra_info", index, {}) or {}
        reward_model = metadata("reward_model", index, {}) or {}
        token_ids = responses[index][mask[index]].tolist()
        records.append({
            "step": int(step),
            "rollout_sample_id": index,
            "uid": str(metadata("uid", index, index)),
            "question_id": extra.get("question_id"),
            "question": extra.get("question"),
            "image_path": extra.get("image_path"),
            "ground_truth": reward_model.get("ground_truth"),
            "response_token_ids": token_ids,
            "response_positions": mask[index].nonzero().flatten().tolist(),
            "output": tokenizer.decode(token_ids, skip_special_tokens=True),
            "score": float(rewards[index]),
            "evidence_available": bool(evidence[index]),
            "teacher_prompt": metadata("teacher_prompt", index, []),
            "teacher_images": metadata("groove_teacher_images", index, []),
        })

    tensor_path = destination / f"{step}.rank0.pt"
    jsonl_path = destination / f"{step}.jsonl"
    tensor_temp = tensor_path.with_suffix(".pt.tmp")
    jsonl_temp = jsonl_path.with_suffix(".jsonl.tmp")
    torch.save(payload, tensor_temp)
    with jsonl_temp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    os.replace(jsonl_temp, jsonl_path)
    os.replace(tensor_temp, tensor_path)
    return int(mask.sum())
