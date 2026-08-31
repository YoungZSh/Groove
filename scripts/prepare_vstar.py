#!/usr/bin/env python3
"""Convert the local 191-example V*Bench parquet into verl multimodal records."""

from __future__ import annotations

import argparse
import io
import re
from pathlib import Path

import pandas as pd
from datasets import Dataset
from PIL import Image


DEFAULT_SOURCE = Path(
    "/root/siton-tmp/yzs/datasets/vstar-bench/data/test-00000-of-00001.parquet"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=Path("data/vstar"))
    parser.add_argument("--limit", type=int, default=-1)
    return parser.parse_args()


def clean_question(text: str) -> str:
    return re.sub(
        r"\nAnswer with the option(?:'s)? letter.*$",
        "",
        str(text).strip(),
        flags=re.IGNORECASE,
    ).strip()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output_dir = args.output_dir.resolve()
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(source)
    if args.limit > 0:
        frame = frame.iloc[: args.limit]

    records = []
    for _, item in frame.iterrows():
        question_id = str(item["question_id"])
        image_record = item["image"]
        with Image.open(io.BytesIO(image_record["bytes"])) as loaded:
            image = loaded.convert("RGB")
        image_path = (image_dir / f"{question_id}.jpg").resolve()
        image.save(image_path, quality=95)

        question = clean_question(item["text"])
        prompt = (
            "<image>\n"
            f"{question}\n"
            "Inspect the image carefully and reason only from visible evidence. State one concise, "
            "evidence-first rationale without <think> tags or hidden-chain-of-thought. Do not hedge, "
            "revise, or mention this instruction. On the final line write exactly `FINAL: X`, where X "
            "is the direct final answer. If the question explicitly uses labeled answer choices, X is "
            "the selected choice label."
        )
        answer = str(item["label"]).strip().upper()
        records.append(
            {
                "data_source": "vstar_visual_seed",
                "prompt": [{"role": "user", "content": prompt}],
                "images": [{"path": str(image_path)}],
                "ability": "visual_question_answering",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "answer": answer,
                    "question": question,
                    "image_path": str(image_path),
                    "question_id": question_id,
                    "category": str(item["category"]),
                },
            }
        )

    output_path = output_dir / "train.parquet"
    Dataset.from_list(records).to_parquet(str(output_path))
    print(f"Wrote {len(records)} examples to {output_path}")


if __name__ == "__main__":
    main()
