"""Deterministic full-frame visual focus for the privileged Teacher branch."""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

from .schemas import EvidenceImageConfig, FocusImage, ObjectCrop


def build_focus_image(
    image_path: Path,
    selected: list[ObjectCrop],
    output_dir: Path,
    config: EvidenceImageConfig,
) -> FocusImage:
    """Keep the union of selected boxes sharp, blend the exterior, then outline.

    Crop boxes retain the same context margin as the crop mode. For an instance
    overlay, use its individual boxes, never the overlay's full-frame extent.
    All pixels are read from the original, not from resized/compressed crops.
    """
    with Image.open(image_path) as loaded:
        original = loaded.convert("RGB")
    boxes = []
    for evidence in selected:
        candidates = (
            [instance.bbox for instance in evidence.instances]
            if evidence.kind == "instance_boxes" else [evidence.expanded_box]
        )
        for x1, y1, x2, y2 in candidates:
            if not all(math.isfinite(value) for value in (x1, y1, x2, y2)) or x1 >= x2 or y1 >= y2:
                raise ValueError("focus boxes must have finite coordinates and positive area")
            box = (
                max(0, math.floor(x1)), max(0, math.floor(y1)),
                min(original.width, math.ceil(x2)), min(original.height, math.ceil(y2)),
            )
            if box[0] < box[2] and box[1] < box[3]:
                boxes.append(box)
    if not boxes:
        raise ValueError("focus evidence requires at least one visible selected box")

    mask = Image.new("L", original.size, 0)
    mask_draw = ImageDraw.Draw(mask)
    # PIL rectangles include both endpoints; project XYXY boxes exclude x2/y2.
    rectangles = [(x1, y1, x2 - 1, y2 - 1) for x1, y1, x2, y2 in boxes]
    for rectangle in rectangles:
        mask_draw.rectangle(rectangle, fill=255)
    blurred = original.filter(ImageFilter.GaussianBlur(radius=config.blur_radius))
    background = Image.blend(original, blurred, config.blur_alpha)
    focused = Image.composite(original, background, mask)
    draw = ImageDraw.Draw(focused)
    line_width = max(1, round(min(original.size) / 256))
    for rectangle in rectangles:
        draw.rectangle(rectangle, outline=(255, 0, 0), width=line_width)

    path = output_dir / "focus.png"
    if path.resolve() == image_path.resolve():
        raise ValueError("focus evidence must not overwrite the original image")
    output_dir.mkdir(parents=True, exist_ok=True)
    # Lossless storage preserves the original pixels inside the boxes, except
    # for the explicitly requested red outlines. No resizing is done here.
    focused.save(path)
    return FocusImage(path=path.resolve(), boxes=boxes)
