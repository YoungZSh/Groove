"""Fixed postprocessing for an on-demand counting tool, separate from crop selection."""
from __future__ import annotations

from dataclasses import dataclass
import math

from .schemas import InstanceBox


@dataclass(frozen=True)
class CountingProfile:
    # Fixed baseline settings, never adjusted to a requested/known object count.
    box_threshold: float = 0.35
    text_threshold: float = 0.25
    # Occluded objects can have heavily overlapping boxes. Keep the native
    # detector set by default; overlap alone does not establish duplication.
    nms_iou_threshold: float | None = None


DEFAULT_COUNTING_PROFILE = CountingProfile()


def counting_region(region, image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = image_size
    if region is None:
        return 0, 0, width, height
    if not isinstance(region, (list, tuple)) or len(region) != 4:
        raise ValueError("region must be [x1, y1, x2, y2] in original-image pixels")
    values = [float(v) for v in region]
    if not all(math.isfinite(v) for v in values):
        raise ValueError("region coordinates must be finite")
    x1, y1, x2, y2 = values
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(
            f"region must use original-image pixels within x=[0,{width}], y=[0,{height}], "
            "with positive area; do not supply coordinates for a resized image"
        )
    return math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2)


def box_iou(a, b) -> float:
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / (area_a + area_b - intersection)


def select_counting_instances(instances: list[InstanceBox], profile=DEFAULT_COUNTING_PROFILE) -> list[int]:
    """Return kept indices using fixed score filtering, preserving overlaps.

    NMS is available only through an explicit profile for offline comparisons.
    It is disabled in the counting tool's default profile and never applies to
    Analyzer candidate images or ground_image/ground_instances outputs.
    """
    if not 0 <= profile.box_threshold <= 1 or (
        profile.nms_iou_threshold is not None and not 0 <= profile.nms_iou_threshold <= 1
    ):
        raise ValueError("counting thresholds must be in [0, 1]")
    order = sorted((i for i, item in enumerate(instances) if item.score >= profile.box_threshold),
                   key=lambda i: instances[i].score, reverse=True)
    if profile.nms_iou_threshold is None:
        return order
    kept = []
    for index in order:
        if all(box_iou(instances[index].bbox, instances[other].bbox) <= profile.nms_iou_threshold for other in kept):
            kept.append(index)
    return kept
