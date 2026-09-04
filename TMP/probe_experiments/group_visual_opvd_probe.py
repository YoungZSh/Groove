#!/usr/bin/env python3
"""Group-contrastive multimodal-skill OPVD probe on V*Bench.

The pipeline is deliberately split into resumable stages:

1. ``rollout``: sample eight visible-reasoning trajectories per question.
2. ``select``: keep reward-diverse groups (the GRPO-like signal groups).
3. ``analyze``: ask an OpenAI-compatible external VLM to contrast the group.
4. ``ground``: execute its grounding program with Grounding DINO and crop.
5. ``probe``: re-score the same rollout tokens under plain and privileged
   visual-skill prefixes, without updating model weights.

Qwen explicit thinking mode is disabled throughout.  The rollout instruction
asks for short visible reasoning in the ordinary assistant response instead.
API credentials are read from environment variables and are never serialized.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import re
import statistics
import sys
import time
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = Path("/root/siton-tmp/yzs/ckpts/Qwen3.5-4B")
DEFAULT_DATA = Path(
    "/root/siton-tmp/yzs/datasets/vstar-bench/data/test-00000-of-00001.parquet"
)
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "group-visual-opvd-probe"
# Despite the HF suffix "base", this is the public Swin-B checkpoint.  Newer
# Grounding DINO 1.6 Pro / DINO-X Pro weights are cloud-API-only.
DEFAULT_GROUNDER = "IDEA-Research/grounding-dino-base"
DEFAULT_ANALYZER_MODEL = "gpt-5.6"
ROLLOUTS_PER_GROUP = 8
KNOWN_SELECTOR_TYPES = {
    "none",
    "ordinal_x",
    "ordinal_y",
    "nearest",
    "overlap",
    "per_query",
    "union",
}

ANALYZER_SYSTEM_PROMPT = """You are a group-contrastive visual hindsight analyzer.

You will receive one image, one multiple-choice visual question, and eight
on-policy rollout trajectories from the same vision-language model. Each
trajectory includes its sampled answer and a binary terminal reward. You do
not receive the ground-truth answer.

Your job is NOT to solve the question or vote on the answer. Compare successful
and failed trajectories and identify the smallest visual evidence region that
best explains their difference: what successful trajectories inspected
correctly, and what failed trajectories missed, confused, or replaced with a
distractor.

Return one executable group-level visual focus program. Grounding DINO will
execute object noun phrases; deterministic code will execute spatial relations.

Critical constraints:
- Do not state the answer, option letter, final attribute value, or conclusion
  in visible_focus_instruction.
- Keep visible_focus_instruction as an operation, e.g. "inspect and compare the
  small emblems on the sails".
- Put exact object phrases only in grounding_queries.
- Use at most three short grounding queries.
- Each selected evidence object is cropped and zoomed independently. Never ask
  for one large union crop spanning multiple objects.
- Use one grounding query per necessary object and at most three objects.
- Base the analysis on visual evidence and rollout differences, not majority
  voting.
- Return ONLY valid JSON matching the requested schema.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    rollout = subparsers.add_parser("rollout", help="Sample eight rollouts per V* question")
    rollout.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    rollout.add_argument("--data", type=Path, default=DEFAULT_DATA)
    rollout.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR / "groups.jsonl")
    rollout.add_argument("--device", default="cuda:0")
    rollout.add_argument("--indices", type=int, nargs="+")
    rollout.add_argument("--limit", type=int)
    rollout.add_argument("--samples-per-question", type=int, default=ROLLOUTS_PER_GROUP)
    rollout.add_argument(
        "--max-sample-attempts",
        type=int,
        help="Finite retry budget for obtaining parseable FINAL labels; defaults to 3x samples.",
    )
    rollout.add_argument("--max-new-tokens", type=int, default=512)
    rollout.add_argument("--temperature", type=float, default=0.8)
    rollout.add_argument("--top-p", type=float, default=0.95)
    rollout.add_argument("--top-k", type=int, default=20)
    rollout.add_argument("--seed", type=int, default=20260824)
    rollout.add_argument(
        "--attn-implementation",
        default="flash_attention_2",
        choices=("flash_attention_2", "sdpa", "eager"),
    )
    rollout.add_argument("--overwrite", action="store_true")

    select = subparsers.add_parser("select", help="Select reward-diverse GRPO-like groups")
    select.add_argument(
        "--groups",
        type=Path,
        nargs="+",
        default=[DEFAULT_OUTPUT_DIR / "groups.jsonl"],
        help="One or more rollout JSONL shards (for example, one per GPU).",
    )
    select.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR / "selected-groups.jsonl")
    select.add_argument(
        "--mode",
        choices=("mixed", "nonuniform", "all"),
        default="mixed",
        help="mixed requires both reward 0 and 1; nonuniform requires >=2 predicted labels.",
    )
    select.add_argument("--min-valid-rollouts", type=int, default=ROLLOUTS_PER_GROUP)
    select.add_argument("--max-groups", type=int)

    analyze = subparsers.add_parser("analyze", help="Run the external group VLM Analyzer")
    analyze.add_argument("--data", type=Path, default=DEFAULT_DATA)
    analyze.add_argument("--groups", type=Path, default=DEFAULT_OUTPUT_DIR / "selected-groups.jsonl")
    analyze.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR / "analyses.jsonl")
    analyze.add_argument("--request-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "analyzer-requests")
    analyze.add_argument(
        "--analyzer-model",
        default=os.environ.get("ANALYZER_MODEL", DEFAULT_ANALYZER_MODEL),
    )
    analyze.add_argument(
        "--base-url",
        default=os.environ.get("ANALYZER_BASE_URL"),
        help="OpenAI-compatible base URL; defaults to ANALYZER_BASE_URL.",
    )
    analyze.add_argument(
        "--api-key-env",
        default="ANALYZER_API_KEY",
        help="Name of the environment variable containing the API key.",
    )
    analyze.add_argument("--max-completion-tokens", type=int, default=2048)
    analyze.add_argument("--timeout", type=float, default=180.0)
    analyze.add_argument("--image-max-side", type=int, default=1600)
    analyze.add_argument(
        "--local-analyzer-model",
        type=Path,
        help="Use a local Qwen3.5 VLM as a temporary Analyzer instead of an API.",
    )
    analyze.add_argument("--device", default="cuda:0")
    analyze.add_argument(
        "--attn-implementation",
        default="flash_attention_2",
        choices=("flash_attention_2", "sdpa", "eager"),
    )
    analyze.add_argument("--no-response-format", action="store_true")
    analyze.add_argument(
        "--dry-run",
        action="store_true",
        help="Write redacted request previews without calling the API.",
    )
    analyze.add_argument("--overwrite", action="store_true")

    ground = subparsers.add_parser("ground", help="Execute analyses with Grounding DINO")
    ground.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ground.add_argument("--analyses", type=Path, default=DEFAULT_OUTPUT_DIR / "analyses.jsonl")
    ground.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR / "grounded.jsonl")
    ground.add_argument("--crop-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "crops")
    ground.add_argument("--grounder-model", default=DEFAULT_GROUNDER)
    ground.add_argument("--device", default="cuda:0")
    ground.add_argument("--box-threshold", type=float, default=0.25)
    ground.add_argument("--text-threshold", type=float, default=0.20)
    ground.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    ground.add_argument("--overwrite", action="store_true")

    probe = subparsers.add_parser("probe", help="Measure prefix visual-skill OPVD signals")
    probe.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    probe.add_argument("--data", type=Path, default=DEFAULT_DATA)
    probe.add_argument("--groups", type=Path, default=DEFAULT_OUTPUT_DIR / "selected-groups.jsonl")
    probe.add_argument("--grounded", type=Path, default=DEFAULT_OUTPUT_DIR / "grounded.jsonl")
    probe.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "probe")
    probe.add_argument("--device", default="cuda:0")
    probe.add_argument("--indices", type=int, nargs="+")
    probe.add_argument("--beta", type=float, default=5.0)
    probe.add_argument("--top-k", type=int, default=20)
    probe.add_argument("--seed", type=int, default=20260824)
    probe.add_argument(
        "--attn-implementation",
        default="flash_attention_2",
        choices=("flash_attention_2", "sdpa", "eager"),
    )
    probe.add_argument("--overwrite", action="store_true")

    trajectory_options = subparsers.add_parser(
        "trajectory-options",
        help="Measure answer probabilities after each rollout's reasoning prefix",
    )
    trajectory_options.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    trajectory_options.add_argument("--data", type=Path, default=DEFAULT_DATA)
    trajectory_options.add_argument(
        "--groups", type=Path, default=DEFAULT_OUTPUT_DIR / "selected-groups.jsonl"
    )
    trajectory_options.add_argument(
        "--grounded", type=Path, default=DEFAULT_OUTPUT_DIR / "grounded.jsonl"
    )
    trajectory_options.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "trajectory-options"
    )
    trajectory_options.add_argument("--device", default="cuda:0")
    trajectory_options.add_argument("--indices", type=int, nargs="+")
    trajectory_options.add_argument("--seed", type=int, default=20260824)
    trajectory_options.add_argument(
        "--attn-implementation",
        default="flash_attention_2",
        choices=("flash_attention_2", "sdpa", "eager"),
    )
    trajectory_options.add_argument("--overwrite", action="store_true")

    trajectory_summary = subparsers.add_parser(
        "summarize-trajectory-options",
        help="Aggregate rollout-conditioned answer probability records",
    )
    trajectory_summary.add_argument("--records", type=Path, nargs="+", required=True)
    trajectory_summary.add_argument("--output", type=Path, required=True)

    summarize = subparsers.add_parser("summarize", help="Rebuild the probe summary")
    summarize.add_argument(
        "--records",
        type=Path,
        nargs="+",
        default=[DEFAULT_OUTPUT_DIR / "probe" / "records.jsonl"],
    )
    summarize.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR / "probe" / "summary.json")

    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return records


def completed_indices(path: Path) -> set[int]:
    if not path.exists():
        return set()
    return {
        int(record["index"])
        for record in read_jsonl(path)
        if "error" not in record and not record.get("dry_run")
    }


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pq.read_table(path).to_pylist()


def load_rgb(row: dict[str, Any]) -> Image.Image:
    return Image.open(BytesIO(row["image"]["bytes"])).convert("RGB")


def rollout_question(question: str) -> str:
    """Replace V*'s direct-answer suffix with a visible-reasoning instruction."""
    cleaned = re.sub(
        r"\n?Answer with the option's letter from the given choices directly\.?\s*$",
        "",
        question.strip(),
        flags=re.IGNORECASE,
    )
    return (
        cleaned
        + "\n\nExplain the relevant visual evidence in 2-5 concise sentences. "
        + "Do not use hidden thinking tags. End exactly with `FINAL: X`, where X is the option letter."
    )


def extract_label(text: str) -> str | None:
    patterns = (
        r"FINAL\s*:\s*\(?([A-D])\)?",
        r"(?:correct\s+answer|answer)\s*(?:is|:)?\s*\(?([A-D])\)?",
        r"\(([A-D])\)[^\n]{0,80}$",
        r"\b([A-D])\s*$",
    )
    for pattern in patterns:
        matches = list(re.finditer(pattern, text, flags=re.IGNORECASE | re.MULTILINE))
        if matches:
            return matches[-1].group(1).upper()
    return None


def option_map(question: str) -> dict[str, str]:
    return {
        label.upper(): value.strip()
        for label, value in re.findall(r"^\(([A-Z])\)\s*(.+)$", question, re.MULTILINE)
    }


def label_entropy(labels: Sequence[str]) -> float:
    if not labels:
        return 0.0
    counts = Counter(labels)
    total = len(labels)
    return float(-sum((count / total) * math.log(count / total) for count in counts.values()))


def group_stats(rollouts: Sequence[dict[str, Any]]) -> dict[str, Any]:
    valid = [record for record in rollouts if record.get("predicted_label")]
    rewards = [int(record["reward"]) for record in valid]
    labels = [str(record["predicted_label"]) for record in valid]
    return {
        "rollouts": len(rollouts),
        "valid_rollouts": len(valid),
        "successes": sum(rewards),
        "failures": len(rewards) - sum(rewards),
        "reward_mean": float(np.mean(rewards)) if rewards else None,
        "reward_std": float(np.std(rewards)) if rewards else None,
        "unique_predictions": len(set(labels)),
        "prediction_counts": dict(Counter(labels)),
        "prediction_entropy_nats": label_entropy(labels),
        "mixed_outcome": bool(rewards and 0 < sum(rewards) < len(rewards)),
    }


def selected_group(
    group: dict[str, Any], mode: str, min_valid_rollouts: int
) -> tuple[bool, str]:
    stats = group.get("stats") or group_stats(group["rollouts"])
    if int(stats["valid_rollouts"]) < min_valid_rollouts:
        return False, "insufficient_valid_rollouts"
    if mode == "mixed":
        return bool(stats["mixed_outcome"]), "mixed_outcome" if stats["mixed_outcome"] else "reward_tied"
    if mode == "nonuniform":
        selected = int(stats["unique_predictions"]) >= 2
        return selected, "prediction_disagreement" if selected else "uniform_prediction"
    return True, "all"


def image_data_url(image: Image.Image, max_side: int = 1600, quality: int = 90) -> str:
    resized = image.copy()
    resized.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buffer = BytesIO()
    resized.save(buffer, format="JPEG", quality=quality, optimize=True)
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def strip_json_fence(text: str) -> str:
    value = text.strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", value, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        value = fence.group(1).strip()
    if not value.startswith("{"):
        start, end = value.find("{"), value.rfind("}")
        if start >= 0 and end > start:
            value = value[start : end + 1]
    return value


def validate_analysis(value: dict[str, Any]) -> dict[str, Any]:
    required_strings = (
        "group_summary",
        "success_common_evidence",
        "failure_common_pattern",
        "visible_focus_instruction",
    )
    normalized = dict(value)
    for key in required_strings:
        text = str(normalized.get(key, "")).strip()
        if not text:
            raise ValueError(f"Analyzer response missing non-empty {key}")
        normalized[key] = text

    queries = normalized.get("grounding_queries")
    if not isinstance(queries, list):
        raise ValueError("grounding_queries must be a list")
    queries = [str(query).strip() for query in queries if str(query).strip()]
    if not 1 <= len(queries) <= 3:
        raise ValueError("grounding_queries must contain 1-3 non-empty queries")
    normalized["grounding_queries"] = queries

    selector = normalized.get("spatial_selector", {"type": "none", "arguments": {}})
    if not isinstance(selector, dict):
        raise ValueError("spatial_selector must be an object")
    selector_type = str(selector.get("type", "none")).strip().lower()
    if selector_type not in KNOWN_SELECTOR_TYPES:
        raise ValueError(f"Unsupported spatial_selector type: {selector_type}")
    arguments = selector.get("arguments") or {}
    if not isinstance(arguments, dict):
        raise ValueError("spatial_selector.arguments must be an object")
    normalized["spatial_selector"] = {"type": selector_type, "arguments": arguments}

    policy = normalized.get("crop_policy") or {}
    if not isinstance(policy, dict):
        raise ValueError("crop_policy must be an object")
    margin = float(policy.get("context_margin", 0.25))
    max_crops = int(policy.get("max_crops", len(queries)))
    if not 0.0 <= margin <= 1.0:
        raise ValueError("crop_policy.context_margin must be in [0, 1]")
    if max_crops not in (1, 2, 3):
        raise ValueError("crop_policy.max_crops must be 1, 2, or 3")
    normalized["crop_policy"] = {
        "context_margin": margin,
        "max_crops": max_crops,
        "preserve_relation_context": bool(policy.get("preserve_relation_context", True)),
    }

    normalized["supporting_success_ids"] = [
        int(item) for item in normalized.get("supporting_success_ids", [])
    ]
    normalized["contrasting_failure_ids"] = [
        int(item) for item in normalized.get("contrasting_failure_ids", [])
    ]
    normalized["confidence"] = float(normalized.get("confidence", 0.0))
    return normalized


def parse_analysis_response(text: str) -> dict[str, Any]:
    return validate_analysis(json.loads(strip_json_fence(text)))


def analyzer_user_text(group: dict[str, Any]) -> str:
    trajectories: list[str] = []
    for rollout in group["rollouts"]:
        trajectories.append(
            "\n".join(
                [
                    f"Trajectory {rollout['rollout_id']}",
                    f"terminal_reward: {rollout.get('reward')}",
                    f"sampled_answer: {rollout.get('predicted_label')}",
                    "reasoning_and_answer:",
                    str(rollout.get("completion", "")).strip(),
                ]
            )
        )
    schema = {
        "group_summary": "string",
        "success_common_evidence": "string",
        "failure_common_pattern": "string",
        "visible_focus_instruction": "operation-only string with no answer leakage",
        "grounding_queries": ["short object noun phrase"],
        "spatial_selector": {
            "type": "none | ordinal_x | ordinal_y | nearest | overlap | per_query",
            "arguments": {},
        },
        "crop_policy": {
            "context_margin": 0.25,
            "max_crops": "number of separately cropped evidence objects, 1-3",
            "preserve_relation_context": True,
        },
        "supporting_success_ids": [0],
        "contrasting_failure_ids": [1],
        "confidence": 0.0,
    }
    return (
        f"Question:\n{group['question']}\n\n"
        + "On-policy rollout group:\n\n"
        + "\n\n---\n\n".join(trajectories)
        + "\n\nReturn exactly this JSON schema:\n"
        + json.dumps(schema, ensure_ascii=False, indent=2)
    )


def analysis_leakage_flags(
    analysis: dict[str, Any], question: str, ground_truth_label: str
) -> list[str]:
    visible = analysis["visible_focus_instruction"].lower()
    flags: list[str] = []
    if re.search(r"\b(?:answer|correct option|final)\b", visible):
        flags.append("answer_language")
    gold = re.escape(ground_truth_label.lower())
    if re.search(
        rf"(?:\boption\s+(?:is\s+)?{gold}\b|\({gold}\)|\banswer\s+(?:is\s+)?{gold}\b)",
        visible,
    ):
        flags.append("gold_option_letter")
    answer_text = option_map(question).get(ground_truth_label, "").strip().lower().rstrip(".")
    if len(answer_text) >= 3 and answer_text in visible:
        flags.append("gold_option_text")
    return flags


def load_qwen(model_path: Path, device: str, attn_implementation: str) -> tuple[Any, Any]:
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
        local_files_only=True,
    ).eval()
    model.to(device)
    return processor, model


def build_messages(
    question: str,
    *,
    focus_instruction: str | None = None,
    evidence_count: int = 0,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {"type": "image"},
        {"type": "text", "text": question},
    ]
    if focus_instruction:
        content.append(
            {
                "type": "text",
                "text": (
                    "\nHindsight visual focus (training-time privileged context):\n"
                    + focus_instruction.strip()
                ),
            }
        )
    for evidence_index in range(evidence_count):
        content.extend(
            [
                {
                    "type": "text",
                    "text": f"\nZoomed visual evidence {evidence_index + 1}:",
                },
                {"type": "image"},
            ]
        )
    return [{"role": "user", "content": content}]


def run_rollout(args: argparse.Namespace) -> int:
    if args.samples_per_question < 2:
        raise ValueError("--samples-per-question must be at least 2")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive for diverse rollouts")
    if args.overwrite and args.output.exists():
        args.output.unlink()

    rows = load_rows(args.data)
    done = completed_indices(args.output)
    indices = args.indices if args.indices is not None else list(range(len(rows)))
    indices = [index for index in indices if 0 <= index < len(rows) and index not in done]
    if args.limit is not None:
        indices = indices[: args.limit]
    if not indices:
        print("No pending questions.")
        return 0

    processor, model = load_qwen(args.model, args.device, args.attn_implementation)
    eos_value = model.generation_config.eos_token_id
    eos_ids = (
        {int(eos_value)}
        if isinstance(eos_value, int)
        else {int(value) for value in (eos_value or [])}
    )

    print(
        f"rollout questions={len(indices)} samples_per_question={args.samples_per_question} "
        f"device={args.device} thinking=false",
        flush=True,
    )
    for group_position, index in enumerate(indices, start=1):
        row = rows[index]
        image = load_rgb(row)
        question = rollout_question(row["text"])
        messages = build_messages(question)
        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        rollouts: list[dict[str, Any]] = []
        rejected_rollouts: list[dict[str, Any]] = []
        max_attempts = args.max_sample_attempts or args.samples_per_question * 3
        if max_attempts < args.samples_per_question:
            raise ValueError("--max-sample-attempts cannot be smaller than the group size")
        started = time.perf_counter()
        try:
            for attempt_id in range(max_attempts):
                if len(rollouts) >= args.samples_per_question:
                    break
                sample_seed = args.seed + index * 1009 + attempt_id
                torch.manual_seed(sample_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(sample_seed)
                batch = processor(text=[prompt], images=[image], return_tensors="pt").to(args.device)
                prompt_tokens = int(batch["input_ids"].shape[-1])
                with torch.inference_mode():
                    generated = model.generate(
                        **batch,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=True,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        use_cache=True,
                    )
                token_ids = generated[0, prompt_tokens:]
                ids = token_ids.detach().cpu().tolist()
                completion = processor.decode(token_ids, skip_special_tokens=True).strip()
                raw_completion = processor.decode(token_ids, skip_special_tokens=False)
                predicted = extract_label(completion)
                stopped_on_eos = bool(ids and ids[-1] in eos_ids)
                sampled = {
                    "attempt_id": attempt_id,
                    "seed": sample_seed,
                    "completion": completion,
                    "raw_completion": raw_completion,
                    "generated_token_ids": ids,
                    "predicted_label": predicted,
                    "reward": int(predicted == row["label"]) if predicted else None,
                    "output_tokens": len(ids),
                    "stopped_on_eos": stopped_on_eos,
                    "truncated": len(ids) >= args.max_new_tokens and not stopped_on_eos,
                }
                if predicted is None:
                    rejected_rollouts.append(sampled)
                    continue
                sampled["rollout_id"] = len(rollouts)
                rollouts.append(sampled)

            record = {
                "index": index,
                "question_id": row["question_id"],
                "category": row["category"],
                "image_path": row["image"].get("path"),
                "image_width": image.width,
                "image_height": image.height,
                "question": row["text"],
                "rollout_question": question,
                "ground_truth_label": row["label"],
                "prompt": prompt,
                "rollouts": rollouts,
                "rejected_rollouts": rejected_rollouts,
                "stats": group_stats(rollouts),
                "elapsed_seconds": time.perf_counter() - started,
                "run": {
                    "model": str(args.model),
                    "samples_per_question": args.samples_per_question,
                    "max_sample_attempts": max_attempts,
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                    "seed": args.seed,
                    "enable_thinking": False,
                },
            }
            append_jsonl(args.output, record)
            stats = record["stats"]
            print(
                f"[{group_position}/{len(indices)}] index={index} "
                f"success={stats['successes']}/{stats['valid_rollouts']} "
                f"unique={stats['unique_predictions']} mixed={stats['mixed_outcome']} "
                f"time={record['elapsed_seconds']:.1f}s",
                flush=True,
            )
        except Exception as exc:
            append_jsonl(
                args.output,
                {
                    "index": index,
                    "question_id": row.get("question_id"),
                    "error": f"{type(exc).__name__}: {exc}",
                    "elapsed_seconds": time.perf_counter() - started,
                },
            )
            print(f"index={index} ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
            raise
    return 0


def run_select(args: argparse.Namespace) -> int:
    groups_by_index: dict[int, dict[str, Any]] = {}
    for path in args.groups:
        for record in read_jsonl(path):
            if "error" not in record:
                groups_by_index[int(record["index"])] = record
    groups = list(groups_by_index.values())
    selected: list[dict[str, Any]] = []
    rejected = Counter()
    for group in groups:
        keep, reason = selected_group(group, args.mode, args.min_valid_rollouts)
        if keep:
            copied = dict(group)
            copied["selection"] = {"mode": args.mode, "reason": reason}
            selected.append(copied)
        else:
            rejected[reason] += 1
    selected.sort(
        key=lambda group: (
            -float(group["stats"].get("reward_std") or 0.0),
            -float(group["stats"].get("prediction_entropy_nats") or 0.0),
            int(group["index"]),
        )
    )
    if args.max_groups is not None:
        selected = selected[: args.max_groups]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for group in selected:
            handle.write(json.dumps(group, ensure_ascii=False) + "\n")
    summary = {
        "input_groups": len(groups),
        "selected_groups": len(selected),
        "selected_indices": [int(group["index"]) for group in selected],
        "mode": args.mode,
        "rejected": dict(rejected),
    }
    write_json(args.output.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def extract_openai_message_text(message: Any) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") in {"text", "output_text"}:
                parts.append(str(item.get("text", "")))
            elif hasattr(item, "text"):
                parts.append(str(item.text))
        return "\n".join(parts)
    return str(content)


def create_analyzer_completion(client: Any, request_kwargs: dict[str, Any]) -> Any:
    """Call common OpenAI-compatible gateway variants without hiding real errors."""
    attempts = [dict(request_kwargs)]
    without_response_format = dict(request_kwargs)
    without_response_format.pop("response_format", None)
    if without_response_format != attempts[-1]:
        attempts.append(without_response_format)
    max_tokens_variant = dict(without_response_format)
    if "max_completion_tokens" in max_tokens_variant:
        max_tokens_variant["max_tokens"] = max_tokens_variant.pop(
            "max_completion_tokens"
        )
        attempts.append(max_tokens_variant)

    last_error: Exception | None = None
    for position, attempt in enumerate(attempts):
        try:
            return client.chat.completions.create(**attempt)
        except Exception as exc:
            last_error = exc
            retryable_shape_error = isinstance(exc, TypeError) or getattr(
                exc, "status_code", None
            ) in {400, 404, 422}
            if not retryable_shape_error or position == len(attempts) - 1:
                raise
    assert last_error is not None
    raise last_error


def analyzer_request_preview(group: dict[str, Any], model: str, image: Image.Image) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": ANALYZER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "<redacted-data-url>", "detail": "high"}},
                    {"type": "text", "text": analyzer_user_text(group)},
                ],
            },
        ],
        "image_size": list(image.size),
        "temperature": 0.0,
    }


def run_local_analyzer(
    model: Any,
    processor: Any,
    image: Image.Image,
    group: dict[str, Any],
    device: str,
    max_new_tokens: int,
) -> str:
    messages = [
        {"role": "system", "content": ANALYZER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": analyzer_user_text(group)},
            ],
        },
    ]
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    batch = processor(text=[prompt], images=[image], return_tensors="pt").to(device)
    prompt_tokens = int(batch["input_ids"].shape[-1])
    with torch.inference_mode():
        generated = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    return processor.decode(
        generated[0, prompt_tokens:], skip_special_tokens=True
    ).strip()


def run_analyze(args: argparse.Namespace) -> int:
    if args.overwrite and args.output.exists():
        args.output.unlink()
    rows = load_rows(args.data)
    groups = [record for record in read_jsonl(args.groups) if "error" not in record]
    done = completed_indices(args.output)
    pending = [group for group in groups if int(group["index"]) not in done]
    args.request_dir.mkdir(parents=True, exist_ok=True)

    client = None
    local_processor = None
    local_model = None
    if not args.dry_run and args.local_analyzer_model is not None:
        local_processor, local_model = load_qwen(
            args.local_analyzer_model,
            args.device,
            args.attn_implementation,
        )
    elif not args.dry_run:
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Missing Analyzer API key. Export {args.api_key_env} before running analyze."
            )
        if not args.base_url:
            raise RuntimeError("Missing Analyzer base URL. Set --base-url or ANALYZER_BASE_URL.")
        from openai import OpenAI

        client = OpenAI(
            api_key=api_key,
            base_url=args.base_url,
            timeout=args.timeout,
            max_retries=3,
        )

    backend = "local" if args.local_analyzer_model is not None else "openai-compatible"
    print(
        f"analyze groups={len(pending)} model={args.analyzer_model} "
        f"backend={backend} dry_run={args.dry_run}",
        flush=True,
    )
    for position, group in enumerate(pending, start=1):
        index = int(group["index"])
        row = rows[index]
        image = load_rgb(row)
        preview = analyzer_request_preview(group, args.analyzer_model, image)
        write_json(args.request_dir / f"index-{index:03d}.json", preview)
        if args.dry_run:
            print(f"[{position}/{len(pending)}] index={index} request preview written")
            continue

        started = time.perf_counter()
        try:
            usage = None
            if local_model is not None and local_processor is not None:
                raw_text = run_local_analyzer(
                    local_model,
                    local_processor,
                    image,
                    group,
                    args.device,
                    args.max_completion_tokens,
                )
            else:
                messages = [
                    {"role": "system", "content": ANALYZER_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_data_url(image, args.image_max_side),
                                    "detail": "high",
                                },
                            },
                            {"type": "text", "text": analyzer_user_text(group)},
                        ],
                    },
                ]
                request_kwargs: dict[str, Any] = {
                    "model": args.analyzer_model,
                    "messages": messages,
                    "temperature": 0.0,
                    "max_completion_tokens": args.max_completion_tokens,
                }
                if not args.no_response_format:
                    request_kwargs["response_format"] = {"type": "json_object"}
                response = create_analyzer_completion(client, request_kwargs)
                raw_text = extract_openai_message_text(response.choices[0].message)
                usage = (
                    response.usage.model_dump()
                    if getattr(response, "usage", None)
                    else None
                )
            analysis = parse_analysis_response(raw_text)
            flags = analysis_leakage_flags(analysis, row["text"], row["label"])
            if flags:
                analysis["raw_visible_focus_instruction"] = analysis[
                    "visible_focus_instruction"
                ]
                analysis["visible_focus_instruction"] = (
                    "Inspect the localized target region and compare it with its "
                    "nearby visual context."
                )
            record = {
                "index": index,
                "question_id": row["question_id"],
                "group_stats": group["stats"],
                "analysis": analysis,
                "leakage_flags": flags,
                "raw_response": raw_text,
                "analyzer_model": (
                    str(args.local_analyzer_model)
                    if args.local_analyzer_model is not None
                    else args.analyzer_model
                ),
                "analyzer_backend": backend,
                "elapsed_seconds": time.perf_counter() - started,
                "usage": usage,
            }
            append_jsonl(args.output, record)
            print(
                f"[{position}/{len(pending)}] index={index} "
                f"queries={analysis['grounding_queries']} leakage={flags} "
                f"time={record['elapsed_seconds']:.1f}s",
                flush=True,
            )
        except Exception as exc:
            append_jsonl(
                args.output,
                {
                    "index": index,
                    "question_id": row.get("question_id"),
                    "error": f"{type(exc).__name__}: {exc}",
                    "elapsed_seconds": time.perf_counter() - started,
                },
            )
            print(f"index={index} ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
            raise
    return 0


def box_center(box: Sequence[float]) -> tuple[float, float]:
    return ((float(box[0]) + float(box[2])) / 2, (float(box[1]) + float(box[3])) / 2)


def box_area(box: Sequence[float]) -> float:
    return max(float(box[2]) - float(box[0]), 0.0) * max(
        float(box[3]) - float(box[1]), 0.0
    )


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    intersection = (
        max(min(float(a[2]), float(b[2])) - max(float(a[0]), float(b[0])), 0.0)
        * max(min(float(a[3]), float(b[3])) - max(float(a[1]), float(b[1])), 0.0)
    )
    union = box_area(a) + box_area(b) - intersection
    return intersection / union if union > 0 else 0.0


def union_boxes(boxes: Sequence[Sequence[float]]) -> list[float]:
    if not boxes:
        raise ValueError("Cannot union an empty list of boxes")
    return [
        min(float(box[0]) for box in boxes),
        min(float(box[1]) for box in boxes),
        max(float(box[2]) for box in boxes),
        max(float(box[3]) for box in boxes),
    ]


def expand_box(
    box: Sequence[float], image_size: tuple[int, int], margin: float
) -> list[int]:
    width, height = image_size
    x0, y0, x1, y1 = [float(value) for value in box]
    box_width, box_height = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    x0 -= box_width * margin
    x1 += box_width * margin
    y0 -= box_height * margin
    y1 += box_height * margin
    result = [
        max(0, int(math.floor(x0))),
        max(0, int(math.floor(y0))),
        min(width, int(math.ceil(x1))),
        min(height, int(math.ceil(y1))),
    ]
    if result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError(f"Invalid expanded box: {result}")
    return result


def _query_groups(candidates: Sequence[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for candidate in candidates:
        grouped.setdefault(int(candidate["query_index"]), []).append(candidate)
    for values in grouped.values():
        values.sort(key=lambda item: float(item["score"]), reverse=True)
    return grouped


def select_grounding_boxes(
    candidates: Sequence[dict[str, Any]], selector: dict[str, Any]
) -> list[dict[str, Any]]:
    if not candidates:
        return []
    selector_type = str(selector.get("type", "none"))
    arguments = selector.get("arguments") or {}
    grouped = _query_groups(candidates)

    if selector_type == "none":
        return [max(candidates, key=lambda item: float(item["score"]))]

    if selector_type in {"per_query", "union"}:
        return [values[0] for _, values in sorted(grouped.items()) if values]

    if selector_type in {"ordinal_x", "ordinal_y"}:
        query_index = int(arguments.get("query_index", 0))
        values = list(grouped.get(query_index, []))
        if not values:
            values = list(candidates)
        axis = 0 if selector_type == "ordinal_x" else 1
        direction = str(
            arguments.get(
                "direction",
                "left_to_right" if axis == 0 else "top_to_bottom",
            )
        )
        reverse = direction in {"right_to_left", "bottom_to_top"}
        values.sort(key=lambda item: box_center(item["box"])[axis], reverse=reverse)
        rank = max(int(arguments.get("rank", 1)), 1)
        return [values[min(rank - 1, len(values) - 1)]]

    if selector_type in {"nearest", "overlap"}:
        first = grouped.get(int(arguments.get("first_query_index", 0)), [])
        second = grouped.get(int(arguments.get("second_query_index", 1)), [])
        if not first or not second:
            return [max(candidates, key=lambda item: float(item["score"]))]
        pairs: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
        for item_a in first:
            for item_b in second:
                if selector_type == "nearest":
                    ax, ay = box_center(item_a["box"])
                    bx, by = box_center(item_b["box"])
                    metric = -math.hypot(ax - bx, ay - by)
                else:
                    metric = box_iou(item_a["box"], item_b["box"])
                pairs.append((metric, item_a, item_b))
        _, item_a, item_b = max(pairs, key=lambda item: item[0])
        return [item_a, item_b]

    raise ValueError(f"Unsupported selector: {selector_type}")


def run_grounding_dino(
    model: Any,
    processor: Any,
    image: Image.Image,
    queries: Sequence[str],
    device: str,
    box_threshold: float,
    text_threshold: float,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for query_index, query in enumerate(queries):
        normalized_query = query.strip().rstrip(".") + "."
        batch = processor(images=image, text=normalized_query, return_tensors="pt").to(device)
        with torch.inference_mode():
            outputs = model(**batch)
        result = processor.post_process_grounded_object_detection(
            outputs,
            input_ids=batch.get("input_ids"),
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[(image.height, image.width)],
        )[0]
        boxes = result["boxes"].detach().cpu().tolist()
        scores = result["scores"].detach().cpu().tolist()
        labels = result.get("text_labels") or result.get("labels") or [query] * len(boxes)
        for box, score, label in zip(boxes, scores, labels):
            candidates.append(
                {
                    "query_index": query_index,
                    "query": query,
                    "label": str(label),
                    "score": float(score),
                    "box": [float(value) for value in box],
                }
            )
    return candidates


def save_crop_visualization(
    image: Image.Image,
    candidates: Sequence[dict[str, Any]],
    selected: Sequence[dict[str, Any]],
    expanded_boxes: Sequence[Sequence[int]],
    output: Path,
) -> None:
    from PIL import ImageDraw

    visual = image.copy()
    draw = ImageDraw.Draw(visual)
    selected_ids = {id(item) for item in selected}
    line_width = max(3, min(image.size) // 400)
    for candidate in candidates:
        color = (255, 80, 40) if id(candidate) in selected_ids else (255, 190, 40)
        draw.rectangle(candidate["box"], outline=color, width=line_width)
    for crop_index, expanded_box in enumerate(expanded_boxes, start=1):
        draw.rectangle(expanded_box, outline=(235, 20, 30), width=line_width + 2)
        draw.text(
            (expanded_box[0] + line_width, expanded_box[1] + line_width),
            f"C{crop_index}",
            fill=(235, 20, 30),
        )
    visual.thumbnail((1400, 1000), Image.Resampling.LANCZOS)
    output.parent.mkdir(parents=True, exist_ok=True)
    visual.save(output, quality=90, optimize=True)


def run_ground(args: argparse.Namespace) -> int:
    if args.overwrite and args.output.exists():
        args.output.unlink()
    rows = load_rows(args.data)
    analyses = [record for record in read_jsonl(args.analyses) if "error" not in record]
    done = completed_indices(args.output)
    pending = [record for record in analyses if int(record["index"]) not in done]
    if not pending:
        print("No pending analyses to ground.")
        return 0

    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        args.grounder_model,
        local_files_only=args.local_files_only,
    )
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        args.grounder_model,
        local_files_only=args.local_files_only,
        dtype=torch.float32,
    ).eval()
    model.to(args.device)
    args.crop_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"ground groups={len(pending)} model={args.grounder_model} device={args.device}",
        flush=True,
    )
    for position, record in enumerate(pending, start=1):
        index = int(record["index"])
        row = rows[index]
        image = load_rgb(row)
        analysis = record["analysis"]
        started = time.perf_counter()
        try:
            candidates = run_grounding_dino(
                model,
                processor,
                image,
                analysis["grounding_queries"],
                args.device,
                args.box_threshold,
                args.text_threshold,
            )
            selected = select_grounding_boxes(candidates, analysis["spatial_selector"])
            if not selected:
                raise ValueError("Grounding DINO returned no usable boxes")
            sample_dir = args.crop_dir / f"index-{index:03d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            visual_path = sample_dir / "grounding-visualization.jpg"
            expanded_boxes: list[list[int]] = []
            crop_records: list[dict[str, Any]] = []
            for crop_index, item in enumerate(selected, start=1):
                expanded = expand_box(
                    item["box"],
                    image.size,
                    float(analysis["crop_policy"]["context_margin"]),
                )
                crop = image.crop(tuple(expanded))
                crop_path = sample_dir / f"object-crop-{crop_index:02d}.jpg"
                crop.save(crop_path, quality=95, optimize=True)
                area_fraction = (crop.width * crop.height) / (
                    image.width * image.height
                )
                expanded_boxes.append(expanded)
                crop_records.append(
                    {
                        "crop_index": crop_index,
                        "query_index": int(item["query_index"]),
                        "query": item["query"],
                        "score": float(item["score"]),
                        "raw_box": item["box"],
                        "expanded_box": expanded,
                        "crop_path": str(crop_path.resolve()),
                        "crop_width": crop.width,
                        "crop_height": crop.height,
                        "crop_area_fraction": area_fraction,
                    }
                )
            save_crop_visualization(
                image, candidates, selected, expanded_boxes, visual_path
            )
            total_crop_area_fraction = sum(
                item["crop_area_fraction"] for item in crop_records
            )
            grounded = {
                "index": index,
                "question_id": row["question_id"],
                "analysis": analysis,
                "leakage_flags": record.get("leakage_flags", []),
                "grounder_model": args.grounder_model,
                "box_threshold": args.box_threshold,
                "text_threshold": args.text_threshold,
                "candidates": candidates,
                "selected": selected,
                "crop_mode": "per_object",
                "crops": crop_records,
                "crop_paths": [item["crop_path"] for item in crop_records],
                "expanded_boxes": expanded_boxes,
                "crop_path": crop_records[0]["crop_path"],
                "visualization_path": str(visual_path.resolve()),
                "crop_count": len(crop_records),
                "crop_area_fractions": [
                    item["crop_area_fraction"] for item in crop_records
                ],
                "total_crop_area_fraction": total_crop_area_fraction,
                "crop_area_fraction": total_crop_area_fraction,
                "elapsed_seconds": time.perf_counter() - started,
            }
            append_jsonl(args.output, grounded)
            print(
                f"[{position}/{len(pending)}] index={index} candidates={len(candidates)} "
                f"crops={len(crop_records)} "
                f"areas={[round(value, 3) for value in grounded['crop_area_fractions']]}",
                flush=True,
            )
        except Exception as exc:
            append_jsonl(
                args.output,
                {
                    "index": index,
                    "question_id": row.get("question_id"),
                    "error": f"{type(exc).__name__}: {exc}",
                    "elapsed_seconds": time.perf_counter() - started,
                },
            )
            print(f"index={index} ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
            raise
    return 0


def common_prefix_length(a: Sequence[int], b: Sequence[int]) -> int:
    length = 0
    for left, right in zip(a, b):
        if left != right:
            break
        length += 1
    return length


def common_suffix_length(a: Sequence[int], b: Sequence[int]) -> int:
    length = 0
    for left, right in zip(reversed(a), reversed(b)):
        if left != right:
            break
        length += 1
    return length


def score_completion(
    model: Any,
    processor: Any,
    messages: list[dict[str, Any]],
    images: Sequence[Image.Image],
    completion: str,
    device: str,
) -> dict[str, Any]:
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_batch = processor(text=[prompt], images=list(images), return_tensors="pt")
    full_batch = processor(
        text=[prompt + completion],
        images=list(images),
        return_tensors="pt",
    )
    prompt_ids = prompt_batch["input_ids"][0].tolist()
    full_ids = full_batch["input_ids"][0].tolist()
    prefix = common_prefix_length(prompt_ids, full_ids)
    if prefix < len(prompt_ids) - 2:
        raise ValueError(
            f"Prompt/completion tokenization diverged too early: {prefix}/{len(prompt_ids)}"
        )
    target_start = prefix
    target_ids = full_ids[target_start:]
    if len(target_ids) < 2:
        raise ValueError("Completion is too short for OPVD scoring")
    query_positions = torch.arange(
        target_start - 1,
        len(full_ids) - 1,
        dtype=torch.long,
        device=device,
    )
    full_batch = {key: value.to(device) for key, value in full_batch.items()}
    with torch.inference_mode():
        logits = model(
            **full_batch,
            use_cache=False,
            logits_to_keep=query_positions,
        ).logits[0].float().cpu()
    if logits.shape[0] != len(target_ids):
        raise ValueError(f"Logit/target mismatch: {logits.shape[0]} vs {len(target_ids)}")
    return {
        "prompt": prompt,
        "target_ids": target_ids,
        "logits": logits,
        "dropped_prompt_boundary_tokens": len(prompt_ids) - prefix,
    }


def aligned_distribution_metrics(
    plain: dict[str, Any],
    teacher: dict[str, Any],
    beta: float,
    top_k: int,
) -> dict[str, Any]:
    suffix = common_suffix_length(plain["target_ids"], teacher["target_ids"])
    if suffix < 2:
        raise ValueError("Plain and Teacher completions do not share an aligned token suffix")
    plain_logits = plain["logits"][-suffix:]
    teacher_logits = teacher["logits"][-suffix:]
    target_ids = torch.tensor(plain["target_ids"][-suffix:], dtype=torch.long)

    plain_logp = F.log_softmax(plain_logits, dim=-1)
    teacher_logp = F.log_softmax(teacher_logits, dim=-1)
    plain_p = plain_logp.exp()
    teacher_p = teacher_logp.exp()
    rows = torch.arange(suffix)
    sampled_plain_logp = plain_logp[rows, target_ids]
    sampled_teacher_logp = teacher_logp[rows, target_ids]
    delta = sampled_teacher_logp - sampled_plain_logp
    gate = torch.sigmoid(float(beta) * delta)
    teacher_to_plain_kl = (teacher_p * (teacher_logp - plain_logp)).sum(-1)
    midpoint = 0.5 * (plain_p + teacher_p)
    jsd = 0.5 * (
        (plain_p * (plain_logp - midpoint.clamp_min(1e-30).log())).sum(-1)
        + (teacher_p * (teacher_logp - midpoint.clamp_min(1e-30).log())).sum(-1)
    )
    plain_top = plain_p.topk(top_k, dim=-1).indices
    teacher_mass_on_plain_top = teacher_p.gather(1, plain_top).sum(-1)
    decoded_target_ids = target_ids.tolist()
    return {
        "aligned_tokens": suffix,
        "plain_prefix_tokens_dropped": len(plain["target_ids"]) - suffix,
        "teacher_prefix_tokens_dropped": len(teacher["target_ids"]) - suffix,
        "mean_sampled_logprob_plain": float(sampled_plain_logp.mean()),
        "mean_sampled_logprob_teacher": float(sampled_teacher_logp.mean()),
        "mean_sampled_logprob_delta": float(delta.mean()),
        "median_sampled_logprob_delta": float(delta.median()),
        "positive_delta_fraction": float((delta > 0).float().mean()),
        "negative_delta_fraction": float((delta < 0).float().mean()),
        "gate_mean": float(gate.mean()),
        "gate_std": float(gate.std(unbiased=False)),
        "gate_gt_half_fraction": float((gate > 0.5).float().mean()),
        "teacher_to_plain_kl": float(teacher_to_plain_kl.mean()),
        "jsd": float(jsd.mean()),
        "teacher_mass_on_plain_topk": float(teacher_mass_on_plain_top.mean()),
        "sampled_opvd_term": float((gate * delta).mean()),
        "target_ids": decoded_target_ids,
        "token_sampled_logprob_delta": delta.tolist(),
        "token_gate": gate.tolist(),
        "token_teacher_to_plain_kl": teacher_to_plain_kl.tolist(),
    }


def option_probabilities(
    model: Any,
    processor: Any,
    messages: list[dict[str, Any]],
    images: Sequence[Image.Image],
    option_letters: Sequence[str],
    device: str,
    assistant_prefix: str | None = None,
) -> dict[str, float]:
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    diagnostic = prompt + (
        assistant_prefix
        if assistant_prefix is not None
        else "After examining the evidence, the final option is ("
    )
    batch = processor(text=[diagnostic], images=list(images), return_tensors="pt")
    candidate_ids: list[int] = []
    for letter in option_letters:
        ids = processor.tokenizer(letter, add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            raise ValueError(f"Option letter {letter!r} is not a single token")
        candidate_ids.append(int(ids[0]))
    batch = {key: value.to(device) for key, value in batch.items()}
    with torch.inference_mode():
        logits = model(**batch, use_cache=False, logits_to_keep=1).logits[0, -1].float()
    probabilities = torch.softmax(logits[candidate_ids], dim=0).cpu().tolist()
    return {letter: float(value) for letter, value in zip(option_letters, probabilities)}


def random_same_size_crop(
    image: Image.Image,
    crop_size: tuple[int, int],
    seed: int,
    index: int,
    crop_slot: int = 0,
) -> Image.Image:
    crop_width = min(crop_size[0], image.width)
    crop_height = min(crop_size[1], image.height)
    digest = hashlib.sha256(
        f"{seed}:{index}:random-crop:{crop_slot}".encode()
    ).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    x0 = int(rng.integers(0, image.width - crop_width + 1))
    y0 = int(rng.integers(0, image.height - crop_height + 1))
    return image.crop((x0, y0, x0 + crop_width, y0 + crop_height))


def matched_shuffled_crops(
    source_crops: Sequence[Image.Image], target_sizes: Sequence[tuple[int, int]]
) -> list[Image.Image] | None:
    if not source_crops:
        return None
    return [
        ImageOps.fit(
            source_crops[index % len(source_crops)],
            size,
            method=Image.Resampling.LANCZOS,
        )
        for index, size in enumerate(target_sizes)
    ]


def probe_conditions(
    original: Image.Image,
    crops: Sequence[Image.Image],
    random_crops: Sequence[Image.Image],
    shuffled_crops: Sequence[Image.Image] | None,
    focus: str,
) -> dict[str, tuple[list[dict[str, Any]], list[Image.Image]]]:
    if not crops:
        raise ValueError("At least one evidence crop is required")
    if len(random_crops) != len(crops):
        raise ValueError("Random controls must match the evidence crop count")
    blanks = [Image.new("RGB", crop.size, (235, 235, 235)) for crop in crops]
    evidence_count = len(crops)
    conditions: dict[str, tuple[list[dict[str, Any]], list[Image.Image]]] = {
        "plain": (build_messages(""), [original]),
        "text_only": (build_messages("", focus_instruction=focus), [original]),
        "crop_only": (
            build_messages("", evidence_count=evidence_count),
            [original, *crops],
        ),
        "text_crop": (
            build_messages(
                "", focus_instruction=focus, evidence_count=evidence_count
            ),
            [original, *crops],
        ),
        "blank_crop": (
            build_messages("", evidence_count=evidence_count),
            [original, *blanks],
        ),
        "random_crop": (
            build_messages("", evidence_count=evidence_count),
            [original, *random_crops],
        ),
    }
    if shuffled_crops is not None:
        if len(shuffled_crops) != evidence_count:
            raise ValueError("Shuffled controls must match the evidence crop count")
        conditions["shuffled_crop"] = (
            build_messages("", evidence_count=evidence_count),
            [original, *shuffled_crops],
        )
    return conditions


def replace_condition_question(
    messages: list[dict[str, Any]], question: str
) -> list[dict[str, Any]]:
    copied = json.loads(json.dumps(messages))
    for item in copied[0]["content"]:
        if item.get("type") == "text" and item.get("text") == "":
            item["text"] = question
            break
    return copied


def safe_mean(values: Iterable[float]) -> float | None:
    materialized = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(statistics.mean(materialized)) if materialized else None


def summarize_probe_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    condition_names = sorted(
        {
            condition
            for record in records
            for rollout in record.get("rollouts", [])
            for condition in rollout.get("conditions", {})
        }
    )
    summary: dict[str, Any] = {
        "groups": len(records),
        "indices": [int(record["index"]) for record in records],
        "rollouts": sum(len(record.get("rollouts", [])) for record in records),
        "conditions": {},
        "by_outcome": {},
    }
    for condition in condition_names:
        metrics = [
            rollout["conditions"][condition]
            for record in records
            for rollout in record.get("rollouts", [])
            if condition in rollout.get("conditions", {})
        ]
        summary["conditions"][condition] = {
            "count": len(metrics),
            "mean_sampled_logprob_delta": safe_mean(
                metric["mean_sampled_logprob_delta"] for metric in metrics
            ),
            "positive_delta_fraction": safe_mean(
                metric["positive_delta_fraction"] for metric in metrics
            ),
            "gate_mean": safe_mean(metric["gate_mean"] for metric in metrics),
            "sampled_opvd_term": safe_mean(
                metric["sampled_opvd_term"] for metric in metrics
            ),
            "teacher_to_plain_kl": safe_mean(
                metric["teacher_to_plain_kl"] for metric in metrics
            ),
            "jsd": safe_mean(metric["jsd"] for metric in metrics),
            "teacher_mass_on_plain_topk": safe_mean(
                metric["teacher_mass_on_plain_topk"] for metric in metrics
            ),
        }
        for outcome_name, reward in (("success", 1), ("failure", 0)):
            outcome_metrics = [
                rollout["conditions"][condition]
                for record in records
                for rollout in record.get("rollouts", [])
                if rollout.get("reward") == reward
                and condition in rollout.get("conditions", {})
            ]
            summary["by_outcome"].setdefault(outcome_name, {})[condition] = {
                "count": len(outcome_metrics),
                "mean_sampled_logprob_delta": safe_mean(
                    metric["mean_sampled_logprob_delta"] for metric in outcome_metrics
                ),
                "gate_mean": safe_mean(metric["gate_mean"] for metric in outcome_metrics),
                "sampled_opvd_term": safe_mean(
                    metric["sampled_opvd_term"] for metric in outcome_metrics
                ),
                "teacher_to_plain_kl": safe_mean(
                    metric["teacher_to_plain_kl"] for metric in outcome_metrics
                ),
            }

    summary["group_gold_option_probability"] = {}
    for condition in sorted(
        {
            condition
            for record in records
            for condition in record.get("option_probabilities", {})
        }
    ):
        probabilities = [
            float(record["option_probabilities"][condition][record["ground_truth_label"]])
            for record in records
            if condition in record.get("option_probabilities", {})
        ]
        plain_probabilities = [
            float(record["option_probabilities"]["plain"][record["ground_truth_label"]])
            for record in records
            if condition in record.get("option_probabilities", {})
        ]
        summary["group_gold_option_probability"][condition] = {
            "mean": safe_mean(probabilities),
            "mean_delta_vs_plain": safe_mean(
                value - plain
                for value, plain in zip(probabilities, plain_probabilities)
            ),
        }
    return summary


def run_probe(args: argparse.Namespace) -> int:
    records_path = args.output_dir / "records.jsonl"
    summary_path = args.output_dir / "summary.json"
    if args.overwrite and records_path.exists():
        records_path.unlink()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.data)
    groups = {
        int(record["index"]): record
        for record in read_jsonl(args.groups)
        if "error" not in record
    }
    grounded = {
        int(record["index"]): record
        for record in read_jsonl(args.grounded)
        if "error" not in record
    }
    done = completed_indices(records_path)
    indices = sorted((groups.keys() & grounded.keys()) - done)
    if args.indices is not None:
        requested = set(args.indices)
        indices = [index for index in indices if index in requested]
    if not indices:
        existing = [record for record in read_jsonl(records_path) if "error" not in record] if records_path.exists() else []
        write_json(summary_path, summarize_probe_records(existing))
        print("No pending grounded groups to probe.")
        return 0

    processor, model = load_qwen(args.model, args.device, args.attn_implementation)
    crop_by_index = {
        index: [
            Image.open(path).convert("RGB")
            for path in grounded[index].get(
                "crop_paths", [grounded[index]["crop_path"]]
            )
        ]
        for index in indices
    }
    print(f"probe groups={len(indices)} device={args.device} conditions=prefix", flush=True)

    for position, index in enumerate(indices, start=1):
        row = rows[index]
        group = groups[index]
        grounding = grounded[index]
        original = load_rgb(row)
        crops = crop_by_index[index]
        random_crops = [
            random_same_size_crop(
                original,
                crop.size,
                args.seed,
                index,
                crop_slot,
            )
            for crop_slot, crop in enumerate(crops)
        ]
        shuffled_index = next((candidate for candidate in indices if candidate != index), None)
        shuffled_source = (
            crop_by_index.get(shuffled_index) if shuffled_index is not None else None
        )
        shuffled_crops = matched_shuffled_crops(
            shuffled_source or [], [crop.size for crop in crops]
        )
        focus = str(grounding["analysis"]["visible_focus_instruction"])
        templates = probe_conditions(
            original,
            crops,
            random_crops,
            shuffled_crops,
            focus,
        )
        conditions = {
            name: (replace_condition_question(messages, group["rollout_question"]), images)
            for name, (messages, images) in templates.items()
        }
        option_letters = list(option_map(row["text"]).keys())
        started = time.perf_counter()
        try:
            option_scores = {
                name: option_probabilities(
                    model,
                    processor,
                    messages,
                    images,
                    option_letters,
                    args.device,
                )
                for name, (messages, images) in conditions.items()
            }
            rollout_results: list[dict[str, Any]] = []
            for rollout in group["rollouts"]:
                completion = str(rollout.get("completion", "")).strip()
                if not completion or rollout.get("predicted_label") is None:
                    continue
                plain_messages, plain_images = conditions["plain"]
                plain_score = score_completion(
                    model,
                    processor,
                    plain_messages,
                    plain_images,
                    completion,
                    args.device,
                )
                condition_results: dict[str, Any] = {}
                for condition_name, (messages, images) in conditions.items():
                    if condition_name == "plain":
                        continue
                    teacher_score = score_completion(
                        model,
                        processor,
                        messages,
                        images,
                        completion,
                        args.device,
                    )
                    condition_results[condition_name] = aligned_distribution_metrics(
                        plain_score,
                        teacher_score,
                        args.beta,
                        args.top_k,
                    )
                    del teacher_score
                rollout_results.append(
                    {
                        "rollout_id": int(rollout["rollout_id"]),
                        "predicted_label": rollout["predicted_label"],
                        "reward": rollout["reward"],
                        "completion": completion,
                        "plain_target_tokens": len(plain_score["target_ids"]),
                        "conditions": condition_results,
                    }
                )
                del plain_score

            result = {
                "index": index,
                "question_id": row["question_id"],
                "category": row["category"],
                "question": row["text"],
                "ground_truth_label": row["label"],
                "group_stats": group["stats"],
                "analysis": grounding["analysis"],
                "leakage_flags": grounding.get("leakage_flags", []),
                "crop_mode": grounding.get("crop_mode", "legacy_union"),
                "crop_paths": grounding.get(
                    "crop_paths", [grounding["crop_path"]]
                ),
                "crop_count": len(crops),
                "crop_area_fractions": grounding.get(
                    "crop_area_fractions", [grounding["crop_area_fraction"]]
                ),
                "total_crop_area_fraction": grounding.get(
                    "total_crop_area_fraction", grounding["crop_area_fraction"]
                ),
                "crop_area_fraction": grounding.get(
                    "total_crop_area_fraction", grounding["crop_area_fraction"]
                ),
                "shuffled_crop_index": shuffled_index,
                "option_probabilities": option_scores,
                "rollouts": rollout_results,
                "beta": args.beta,
                "elapsed_seconds": time.perf_counter() - started,
            }
            append_jsonl(records_path, result)
            all_records = [
                record for record in read_jsonl(records_path) if "error" not in record
            ]
            write_json(summary_path, summarize_probe_records(all_records))
            text_crop_delta = safe_mean(
                rollout["conditions"]["text_crop"]["mean_sampled_logprob_delta"]
                for rollout in rollout_results
            )
            gold_delta = (
                option_scores["text_crop"][row["label"]]
                - option_scores["plain"][row["label"]]
            )
            print(
                f"[{position}/{len(indices)}] index={index} rollouts={len(rollout_results)} "
                f"text_crop_delta={text_crop_delta:+.4f} gold_delta={gold_delta:+.4f} "
                f"time={result['elapsed_seconds']:.1f}s",
                flush=True,
            )
        except Exception as exc:
            append_jsonl(
                records_path,
                {
                    "index": index,
                    "question_id": row.get("question_id"),
                    "error": f"{type(exc).__name__}: {exc}",
                    "elapsed_seconds": time.perf_counter() - started,
                },
            )
            print(f"index={index} ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
            raise
    return 0


def reasoning_prefix_before_final(completion: str) -> str:
    matches = list(
        re.finditer(
            r"\bFINAL\s*:\s*\(?[A-D]\)?",
            completion,
            flags=re.IGNORECASE,
        )
    )
    if not matches:
        raise ValueError("Completion has no FINAL marker for trajectory option probing")
    reasoning = completion[: matches[-1].start()].rstrip()
    return reasoning + "\nFINAL: "


def _argmax_label(probabilities: dict[str, float]) -> str:
    return max(probabilities, key=probabilities.__getitem__)


def summarize_trajectory_option_records(
    records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    conditions = sorted(
        {
            condition
            for record in records
            for rollout in record.get("rollouts", [])
            for condition in rollout.get("conditions", {})
        }
    )
    summary: dict[str, Any] = {
        "groups": len(records),
        "indices": [int(record["index"]) for record in records],
        "rollouts": sum(len(record.get("rollouts", [])) for record in records),
        "by_outcome": {},
    }
    for outcome_name, reward in (("success", 1), ("failure", 0)):
        summary["by_outcome"][outcome_name] = {}
        for condition in conditions:
            metrics = [
                rollout["conditions"][condition]
                for record in records
                for rollout in record.get("rollouts", [])
                if rollout.get("reward") == reward
                and condition in rollout.get("conditions", {})
            ]
            count = len(metrics)
            summary["by_outcome"][outcome_name][condition] = {
                "count": count,
                "mean_gold_probability": safe_mean(
                    metric["gold_probability"] for metric in metrics
                ),
                "mean_gold_delta_vs_plain": safe_mean(
                    metric["gold_delta_vs_plain"] for metric in metrics
                ),
                "gold_increase_fraction": (
                    sum(metric["gold_delta_vs_plain"] > 0 for metric in metrics)
                    / count
                    if count
                    else None
                ),
                "mean_sampled_probability": safe_mean(
                    metric["sampled_probability"] for metric in metrics
                ),
                "mean_sampled_delta_vs_plain": safe_mean(
                    metric["sampled_delta_vs_plain"] for metric in metrics
                ),
                "sampled_decrease_fraction": (
                    sum(metric["sampled_delta_vs_plain"] < 0 for metric in metrics)
                    / count
                    if count
                    else None
                ),
                "mean_gold_minus_sampled_margin_delta": safe_mean(
                    metric["margin_delta_vs_plain"] for metric in metrics
                ),
                "desired_direction_fraction": (
                    sum(bool(metric["desired_direction"]) for metric in metrics)
                    / count
                    if count
                    else None
                ),
                "argmax_gold_fraction": (
                    sum(bool(metric["argmax_is_gold"]) for metric in metrics) / count
                    if count
                    else None
                ),
                "corrected_from_plain_fraction": (
                    sum(bool(metric["corrected_from_plain"]) for metric in metrics)
                    / count
                    if count
                    else None
                ),
            }
    return summary


def run_trajectory_options(args: argparse.Namespace) -> int:
    records_path = args.output_dir / "records.jsonl"
    summary_path = args.output_dir / "summary.json"
    if args.overwrite and records_path.exists():
        records_path.unlink()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.data)
    groups = {
        int(record["index"]): record
        for record in read_jsonl(args.groups)
        if "error" not in record
    }
    grounded = {
        int(record["index"]): record
        for record in read_jsonl(args.grounded)
        if "error" not in record
    }
    done = completed_indices(records_path)
    indices = sorted((groups.keys() & grounded.keys()) - done)
    if args.indices is not None:
        requested = set(args.indices)
        indices = [index for index in indices if index in requested]
    if not indices:
        existing = (
            [record for record in read_jsonl(records_path) if "error" not in record]
            if records_path.exists()
            else []
        )
        write_json(summary_path, summarize_trajectory_option_records(existing))
        print("No pending groups for trajectory option probing.")
        return 0

    processor, model = load_qwen(args.model, args.device, args.attn_implementation)
    crop_by_index = {
        index: [
            Image.open(path).convert("RGB")
            for path in grounded[index].get(
                "crop_paths", [grounded[index]["crop_path"]]
            )
        ]
        for index in indices
    }
    print(
        f"trajectory-options groups={len(indices)} device={args.device}",
        flush=True,
    )

    for position, index in enumerate(indices, start=1):
        row = rows[index]
        group = groups[index]
        grounding = grounded[index]
        original = load_rgb(row)
        crops = crop_by_index[index]
        random_crops = [
            random_same_size_crop(
                original, crop.size, args.seed, index, crop_slot
            )
            for crop_slot, crop in enumerate(crops)
        ]
        shuffled_index = next(
            (candidate for candidate in indices if candidate != index), None
        )
        shuffled_source = (
            crop_by_index.get(shuffled_index) if shuffled_index is not None else None
        )
        shuffled_crops = matched_shuffled_crops(
            shuffled_source or [], [crop.size for crop in crops]
        )
        templates = probe_conditions(
            original,
            crops,
            random_crops,
            shuffled_crops,
            str(grounding["analysis"]["visible_focus_instruction"]),
        )
        conditions = {
            name: (
                replace_condition_question(messages, group["rollout_question"]),
                images,
            )
            for name, (messages, images) in templates.items()
        }
        option_letters = list(option_map(row["text"]).keys())
        gold = str(row["label"])
        started = time.perf_counter()
        try:
            rollout_records: list[dict[str, Any]] = []
            for rollout in group["rollouts"]:
                sampled = str(rollout["predicted_label"])
                assistant_prefix = reasoning_prefix_before_final(
                    str(rollout["completion"])
                )
                probabilities = {
                    name: option_probabilities(
                        model,
                        processor,
                        messages,
                        images,
                        option_letters,
                        args.device,
                        assistant_prefix=assistant_prefix,
                    )
                    for name, (messages, images) in conditions.items()
                }
                plain = probabilities["plain"]
                plain_argmax = _argmax_label(plain)
                condition_records: dict[str, Any] = {}
                for name, values in probabilities.items():
                    gold_delta = values[gold] - plain[gold]
                    sampled_delta = values[sampled] - plain[sampled]
                    plain_margin = plain[gold] - plain[sampled]
                    margin = values[gold] - values[sampled]
                    argmax = _argmax_label(values)
                    reward = int(rollout["reward"])
                    condition_records[name] = {
                        "probabilities": values,
                        "gold_probability": values[gold],
                        "sampled_probability": values[sampled],
                        "gold_delta_vs_plain": gold_delta,
                        "sampled_delta_vs_plain": sampled_delta,
                        "gold_minus_sampled_margin": margin,
                        "margin_delta_vs_plain": margin - plain_margin,
                        "argmax": argmax,
                        "argmax_is_gold": argmax == gold,
                        "corrected_from_plain": plain_argmax != gold and argmax == gold,
                        "desired_direction": (
                            gold_delta > 0
                            if reward == 1
                            else gold_delta > 0 and sampled_delta < 0
                        ),
                    }
                rollout_records.append(
                    {
                        "rollout_id": int(rollout["rollout_id"]),
                        "reward": int(rollout["reward"]),
                        "sampled_label": sampled,
                        "gold_label": gold,
                        "reasoning_prefix": assistant_prefix,
                        "plain_argmax": plain_argmax,
                        "conditions": condition_records,
                    }
                )
            result = {
                "index": index,
                "question_id": row["question_id"],
                "question": row["text"],
                "ground_truth_label": gold,
                "crop_mode": grounding.get("crop_mode", "legacy_union"),
                "crop_paths": grounding.get(
                    "crop_paths", [grounding["crop_path"]]
                ),
                "crop_count": len(crops),
                "rollouts": rollout_records,
                "elapsed_seconds": time.perf_counter() - started,
            }
            append_jsonl(records_path, result)
            existing = [
                record for record in read_jsonl(records_path) if "error" not in record
            ]
            write_json(summary_path, summarize_trajectory_option_records(existing))
            correct_metrics = [
                rollout["conditions"]["text_crop"]
                for rollout in rollout_records
                if rollout["reward"] == 1
            ]
            failure_metrics = [
                rollout["conditions"]["text_crop"]
                for rollout in rollout_records
                if rollout["reward"] == 0
            ]
            print(
                f"[{position}/{len(indices)}] index={index} "
                f"success_gold_delta={safe_mean(item['gold_delta_vs_plain'] for item in correct_metrics):+.4f} "
                f"failure_gold_delta={safe_mean(item['gold_delta_vs_plain'] for item in failure_metrics):+.4f} "
                f"failure_wrong_delta={safe_mean(item['sampled_delta_vs_plain'] for item in failure_metrics):+.4f}",
                flush=True,
            )
        except Exception as exc:
            append_jsonl(
                records_path,
                {
                    "index": index,
                    "question_id": row.get("question_id"),
                    "error": f"{type(exc).__name__}: {exc}",
                    "elapsed_seconds": time.perf_counter() - started,
                },
            )
            raise
    return 0


def run_summarize_trajectory_options(args: argparse.Namespace) -> int:
    records = [
        record
        for path in args.records
        for record in read_jsonl(path)
        if "error" not in record
    ]
    summary = summarize_trajectory_option_records(records)
    write_json(args.output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def run_summarize(args: argparse.Namespace) -> int:
    records = [
        record
        for path in args.records
        for record in read_jsonl(path)
        if "error" not in record
    ]
    summary = summarize_probe_records(records)
    write_json(args.output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    args = parse_args()
    if args.command == "rollout":
        return run_rollout(args)
    if args.command == "select":
        return run_select(args)
    if args.command == "analyze":
        return run_analyze(args)
    if args.command == "ground":
        return run_ground(args)
    if args.command == "probe":
        return run_probe(args)
    if args.command == "trajectory-options":
        return run_trajectory_options(args)
    if args.command == "summarize-trajectory-options":
        return run_summarize_trajectory_options(args)
    if args.command == "summarize":
        return run_summarize(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
