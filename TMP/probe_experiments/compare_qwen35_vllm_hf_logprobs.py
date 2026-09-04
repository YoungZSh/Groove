#!/usr/bin/env python3
"""Compare chosen-token log-probs from vLLM and Transformers on fixed text."""

from __future__ import annotations

import argparse
import json
import math
from io import BytesIO
from pathlib import Path

import pyarrow.parquet as pq
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration


DEFAULT_MODEL = Path("/root/siton-tmp/yzs/ckpts/Qwen3.5-2B")
MESSAGES = [
    {
        "role": "system",
        "content": "You are a visual question-answering assistant.",
    },
    {
        "role": "user",
        "content": "Give one short factual sentence about why the sky appears blue.",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["vllm", "hf"], required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--image", action="store_true")
    parser.add_argument(
        "--parquet",
        type=Path,
        default=Path("data/deepeyes_vstar_grpo_2200_seed20260904/train.parquet"),
    )
    parser.add_argument("--row-index", type=int, default=0)
    return parser.parse_args()


def render_prompt(model: Path) -> str:
    processor = AutoProcessor.from_pretrained(model, local_files_only=True)
    return processor.apply_chat_template(
        MESSAGES,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def image_case(args: argparse.Namespace) -> tuple[str, Image.Image]:
    row = pq.read_table(args.parquet).slice(args.row_index, 1).to_pylist()[0]
    image_item = row["images"][0]
    payload = image_item.get("bytes")
    if payload is None:
        payload = Path(image_item["path"]).read_bytes()
    image = Image.open(BytesIO(payload)).convert("RGB")
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    messages = [
        {"role": "system", "content": row["prompt"][0]["content"]},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": row["extra_info"]["question"]},
            ],
        },
    ]
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return prompt, image


def run_vllm(args: argparse.Namespace) -> None:
    from vllm import LLM, SamplingParams

    prompt, image = image_case(args) if args.image else (render_prompt(args.model), None)
    llm = LLM(
        model=str(args.model),
        tensor_parallel_size=1,
        dtype="bfloat16",
        max_model_len=2048,
        max_num_seqs=8,
        max_num_batched_tokens=2048,
        gpu_memory_utilization=0.3,
        enforce_eager=True,
        skip_mm_profiling=True,
        disable_log_stats=True,
        seed=20260904,
    )
    request = {"prompt": prompt, "multi_modal_data": {"image": image}} if image is not None else prompt
    output = llm.generate(
        [request],
        SamplingParams(
            temperature=0.0,
            max_tokens=64,
            logprobs=1,
            seed=20260904,
        ),
        use_tqdm=False,
    )[0]
    item = output.outputs[0]
    chosen_logprobs = [
        float(logprob_map[token_id].logprob)
        for token_id, logprob_map in zip(item.token_ids, item.logprobs, strict=True)
    ]
    trace = {
        "prompt": prompt,
        "prompt_token_ids": list(output.prompt_token_ids),
        "output_text": item.text,
        "output_token_ids": list(item.token_ids),
        "vllm_logprobs": chosen_logprobs,
        "image": args.image,
        "row_index": args.row_index,
    }
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    args.trace.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in trace.items() if k != "prompt"}, ensure_ascii=False))


def run_hf(args: argparse.Namespace) -> None:
    trace = json.loads(args.trace.read_text(encoding="utf-8"))
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    if trace.get("image"):
        prompt, image = image_case(args)
        if prompt != trace["prompt"]:
            raise RuntimeError("rendered image prompt changed between stages")
        model_inputs = processor(text=[prompt], images=[image], return_tensors="pt")
        prompt_ids = model_inputs["input_ids"]
    else:
        model_inputs = {}
        prompt_ids = processor.tokenizer(
            trace["prompt"],
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"]
    expected_prompt_ids = torch.tensor([trace["prompt_token_ids"]], dtype=torch.long)
    if not torch.equal(prompt_ids, expected_prompt_ids):
        raise RuntimeError(
            f"prompt tokenization mismatch: hf={prompt_ids.shape} vllm={expected_prompt_ids.shape}"
        )
    output_ids = torch.tensor([trace["output_token_ids"]], dtype=torch.long)
    input_ids = torch.cat([prompt_ids, output_ids], dim=1).cuda()
    attention_mask = torch.ones_like(input_ids)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).eval().cuda()
    forward_kwargs = {
        key: value.cuda()
        for key, value in model_inputs.items()
        if key not in {"input_ids", "attention_mask", "mm_token_type_ids"}
    }
    if "mm_token_type_ids" in model_inputs:
        suffix = torch.zeros_like(output_ids)
        forward_kwargs["mm_token_type_ids"] = torch.cat(
            [model_inputs["mm_token_type_ids"], suffix], dim=1
        ).cuda()
    with torch.inference_mode():
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            **forward_kwargs,
        ).logits
    start = prompt_ids.shape[1] - 1
    positions = logits[:, start : start + output_ids.shape[1], :].float()
    hf_logprobs = positions.log_softmax(-1).gather(-1, output_ids.cuda().unsqueeze(-1)).squeeze(-1)[0].cpu()
    vllm_logprobs = torch.tensor(trace["vllm_logprobs"], dtype=torch.float32)
    logprob_abs = (hf_logprobs - vllm_logprobs).abs()
    probability_abs = (hf_logprobs.exp() - vllm_logprobs.exp()).abs()
    correlation = torch.corrcoef(torch.stack([hf_logprobs.exp(), vllm_logprobs.exp()]))[0, 1]
    report = {
        "tokens": output_ids.shape[1],
        "output_text": trace["output_text"],
        "hf_logprobs": hf_logprobs.tolist(),
        "vllm_logprobs": vllm_logprobs.tolist(),
        "mean_abs_logprob_diff": float(logprob_abs.mean()),
        "max_abs_logprob_diff": float(logprob_abs.max()),
        "mean_abs_probability_diff": float(probability_abs.mean()),
        "max_abs_probability_diff": float(probability_abs.max()),
        "probability_pearson": float(correlation) if math.isfinite(float(correlation)) else None,
    }
    report_path = args.trace.with_suffix(".comparison.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    if args.stage == "vllm":
        run_vllm(args)
    else:
        run_hf(args)


if __name__ == "__main__":
    main()
