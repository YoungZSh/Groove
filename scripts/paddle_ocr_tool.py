#!/usr/bin/env python3
"""Run the isolated GPU PaddleOCR worker and emit one JSON tool result.

This script intentionally lives outside the student/VERL environment.  The
Analyzer invokes it as a local process; the 4B student model never imports
PaddleOCR or executes tools.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from paddleocr import PaddleOCR
from PIL import Image


def _parse_bbox(value: str | None, image_size: tuple[int, int]) -> tuple[int, int, int, int] | None:
    if value is None:
        return None
    try:
        coords = [float(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("bbox must be x1,y1,x2,y2") from exc
    if len(coords) != 4:
        raise argparse.ArgumentTypeError("bbox must contain four coordinates")
    x1, y1, x2, y2 = coords
    width, height = image_size
    result = (
        max(0, min(width, round(x1))),
        max(0, min(height, round(y1))),
        max(0, min(width, round(x2))),
        max(0, min(height, round(y2))),
    )
    if result[0] >= result[2] or result[1] >= result[3]:
        raise argparse.ArgumentTypeError("bbox must have positive width and height")
    return result


def _json_result(
    ocr: PaddleOCR,
    image_path: Path,
    bbox: tuple[int, int, int, int] | None,
    scale: int,
    min_confidence: float,
) -> dict[str, Any]:
    with Image.open(image_path) as loaded:
        original = loaded.convert("RGB")
    origin_x, origin_y = 0, 0
    working = original
    if bbox is not None:
        origin_x, origin_y, _, _ = bbox
        working = original.crop(bbox)
    if scale > 1:
        working = working.resize(
            (working.width * scale, working.height * scale), Image.Resampling.LANCZOS
        )

    result = ocr.ocr(np.asarray(working), cls=False)
    text_lines: list[dict[str, Any]] = []
    for page in result:
        for line in page or []:
            polygon, (text, confidence) = line
            confidence = float(confidence)
            if confidence < min_confidence:
                continue
            text_lines.append(
                {
                    "text": str(text),
                    "confidence": confidence,
                    "bbox": [
                        [round(float(x) / scale + origin_x, 2), round(float(y) / scale + origin_y, 2)]
                        for x, y in polygon
                    ],
                }
            )
    return {
        "image_path": str(image_path.resolve()),
        "image_size": list(original.size),
        "crop_bbox": list(bbox) if bbox else None,
        "scale": scale,
        "text": text_lines,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--bbox", default=None, help="optional x1,y1,x2,y2 in original-image pixels")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    args = parser.parse_args()

    if args.scale < 1 or args.scale > 8:
        raise ValueError("scale must be in [1, 8]")
    if not 0.0 <= args.min_confidence <= 1.0:
        raise ValueError("min-confidence must be in [0, 1]")

    logging.disable(logging.CRITICAL)
    image_path = Path(args.image).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    with Image.open(image_path) as image:
        bbox = _parse_bbox(args.bbox, image.size)
    # Scene signs are often only a few dozen source pixels high.  Preserve the
    # enlarged crop instead of letting PaddleOCR's default detector cap shrink
    # it back to a document-style short side.
    ocr = PaddleOCR(
        lang="en",
        use_gpu=True,
        use_angle_cls=False,
        show_log=False,
        det_db_box_thresh=0.2,
        det_limit_side_len=4096,
    )
    print(json.dumps(_json_result(ocr, image_path, bbox, args.scale, args.min_confidence), ensure_ascii=False))


if __name__ == "__main__":
    main()
