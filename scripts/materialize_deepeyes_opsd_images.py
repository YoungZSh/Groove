#!/usr/bin/env python3
"""Add local image paths to an existing embedded-image DeepEyes split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/deepeyes_vstar_grpo_2200_seed20260904"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def image_suffix(payload: bytes) -> str:
    if payload.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if payload.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if payload.startswith(b"RIFF") and payload[8:12] == b"WEBP":
        return ".webp"
    return ".img"


def image_payload(image: dict) -> bytes:
    payload = image.get("bytes")
    if payload is not None:
        return payload
    source_path = Path(str(image.get("path", "")))
    if not source_path.is_file():
        raise FileNotFoundError(f"embedded image bytes and readable source path are both missing: {source_path}")
    return source_path.read_bytes()


def convert_split(input_path: Path, output_path: Path, image_dir: Path) -> int:
    records = pq.read_table(input_path).to_pylist()
    converted = []
    for row in records:
        images = row.get("images") or []
        if len(images) != 1:
            raise ValueError(f"expected exactly one source image, got {len(images)}")
        payload = image_payload(images[0])
        digest = hashlib.sha256(payload).hexdigest()
        extra = dict(row.get("extra_info") or {})
        expected_digest = str(extra.get("image_sha256", ""))
        if expected_digest and expected_digest != digest:
            raise ValueError(f"image hash mismatch for {extra.get('question_id')}: {digest}")

        local_path = (image_dir / f"{digest}{image_suffix(payload)}").resolve()
        if local_path.exists():
            if hashlib.sha256(local_path.read_bytes()).hexdigest() != digest:
                raise ValueError(f"existing materialized image has the wrong content: {local_path}")
        else:
            local_path.write_bytes(payload)
        extra["image_path"] = str(local_path)
        row["extra_info"] = extra
        converted.append(row)

    pq.write_table(pa.Table.from_pylist(converted), output_path, compression="zstd")
    return len(converted)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    outputs = [output_dir / "train.parquet", output_dir / "validation.parquet"]
    if any(path.exists() for path in outputs):
        raise FileExistsError(f"refusing to overwrite an existing converted split in {output_dir}")

    counts = {
        split: convert_split(input_dir / f"{split}.parquet", output_dir / f"{split}.parquet", image_dir)
        for split in ("train", "validation")
    }
    source_manifest = json.loads((input_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest = {
        **source_manifest,
        "opsd_materialization": {
            "source_split": str(input_dir),
            "image_dir": str(image_dir.resolve()),
            "records": counts,
            "only_added_extra_info_field": "image_path",
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["opsd_materialization"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
