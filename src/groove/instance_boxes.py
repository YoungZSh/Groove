"""Validate detector candidates and draw boxes without altering scene content."""

from __future__ import annotations

from PIL import Image, ImageDraw

from .schemas import InstanceBox


def normalize_instance_boxes(values: list, image_size: tuple[int, int]) -> list[InstanceBox]:
    """Clip finite boxes to the image; preserve candidate order and duplicates.

    An invalid candidate fails the result instead of silently changing its count.
    This is geometry validation, not a claim that the boxes are correct objects.
    """
    if not isinstance(values, list):
        raise ValueError("instances must be a list")
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    instances = []
    for value in values:
        item = value if isinstance(value, InstanceBox) else InstanceBox.model_validate(value)
        x1, y1, x2, y2 = item.bbox
        bbox = (max(0.0, x1), max(0.0, y1), min(float(width), x2), min(float(height), y2))
        instances.append(InstanceBox(bbox=bbox, score=item.score))
    return instances


def render_instance_boxes(image: Image.Image, instances: list[InstanceBox]) -> Image.Image:
    """Return an RGB copy at native resolution with thin, unfilled red boxes.

    No count, IDs, OCR text, or answer is painted on the Teacher evidence image.
    """
    if not instances:
        raise ValueError("cannot render empty instance-box evidence")
    instances = normalize_instance_boxes(instances, image.size)
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    line_width = max(1, round(min(image.size) / 250))
    for instance in instances:
        x1, y1, x2, y2 = instance.bbox
        left, top = min(image.width - 1, round(x1)), min(image.height - 1, round(y1))
        bounds = (left, top, max(left, min(image.width - 1, round(x2))), max(top, min(image.height - 1, round(y2))))
        draw.rectangle(bounds, outline=(255, 0, 0), width=line_width)
    return annotated
