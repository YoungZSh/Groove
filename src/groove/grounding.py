"""Grounding DINO localization and per-object Crop/Zoom generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from PIL import Image

from .instance_boxes import normalize_instance_boxes, render_instance_boxes
from .schemas import FocusProgram, ObjectCrop, ToolRegion


class Grounder(Protocol):
    def crop_objects(
        self,
        image_path: Path,
        focus: FocusProgram,
        output_dir: Path,
    ) -> list[ObjectCrop]: ...


def expand_box(
    box: tuple[float, float, float, float],
    image_size: tuple[int, int],
    margin: float,
) -> tuple[int, int, int, int]:
    width, height = image_size
    x1, y1, x2, y2 = box
    box_width = max(1.0, x2 - x1)
    box_height = max(1.0, y2 - y1)
    return (
        max(0, int(x1 - box_width * margin)),
        max(0, int(y1 - box_height * margin)),
        min(width, int(x2 + box_width * margin + 0.999)),
        min(height, int(y2 + box_height * margin + 0.999)),
    )


def enlarge_crop(
    image: Image.Image,
    min_short_side: int,
    *,
    max_scale: float = 10.0,
) -> Image.Image:
    """Upscale a crop while preventing extreme interpolation blur.

    Tiny detector boxes can otherwise be enlarged from a few dozen pixels to
    the 768-pixel minimum, creating a visually large but information-free
    Teacher image.  ``max_scale`` is measured against the crop's original
    shortest side, so a 30-pixel box is capped at 300 pixels by default.
    """
    if min_short_side <= 0:
        raise ValueError("min_short_side must be positive")
    if max_scale <= 0:
        raise ValueError("max_scale must be positive")
    short_side = min(image.size)
    if short_side >= min_short_side:
        return image
    scale = min(min_short_side / max(short_side, 1), float(max_scale))
    new_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def crop_tool_regions(
    image_path: Path,
    regions: list[ToolRegion],
    output_dir: Path,
    *,
    min_short_side: int = 768,
    max_scale: float = 10.0,
) -> list[ObjectCrop]:
    """Materialize Analyzer tool regions without rerunning a text-only detector."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as loaded:
        image = loaded.convert("RGB")
    original_area = image.width * image.height
    crops: list[ObjectCrop] = []
    for index, region in enumerate(regions, start=1):
        if region.kind == "instance_boxes":
            instances = normalize_instance_boxes(region.instances, image.size)
            annotated = render_instance_boxes(image, instances)
            path = output_dir / f"tool-boxes-{index:02d}.png"
            if path.resolve() == image_path.resolve():
                raise ValueError("instance evidence must not overwrite the original image")
            annotated.save(path)
            crops.append(ObjectCrop(
                query=region.query, score=region.score,
                raw_box=(0.0, 0.0, float(image.width), float(image.height)),
                expanded_box=(0, 0, image.width, image.height), path=path.resolve(),
                area_fraction=1.0, kind="instance_boxes", instances=instances,
            ))
            continue
        x1, y1, x2, y2 = region.expanded_box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.width, x2), min(image.height, y2)
        if x1 >= x2 or y1 >= y2:
            continue
        crop = image.crop((x1, y1, x2, y2))
        path = output_dir / f"tool-crop-{index:02d}.jpg"
        enlarge_crop(crop, min_short_side, max_scale=max_scale).save(path, quality=95, optimize=True)
        crops.append(
            ObjectCrop(
                query=region.query,
                score=region.score,
                raw_box=(float(x1), float(y1), float(x2), float(y2)),
                expanded_box=(x1, y1, x2, y2),
                path=path.resolve(),
                area_fraction=(crop.width * crop.height) / original_area,
            )
        )
    return crops


@dataclass(frozen=True)
class GroundingDinoConfig:
    model: str = "IDEA-Research/grounding-dino-base"
    device: str = "cpu"
    box_threshold: float = 0.25
    text_threshold: float = 0.20
    min_short_side: int = 768
    max_crop_scale: float = 10.0
    local_files_only: bool = True


class GroundingDinoGrounder:
    def __init__(self, config: GroundingDinoConfig):
        self.config = config
        self._processor = None
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._processor, self._model
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(
            self.config.model,
            local_files_only=self.config.local_files_only,
        )
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.config.model,
            local_files_only=self.config.local_files_only,
        ).to(self.config.device).eval()
        if self.config.device.startswith("cuda"):
            self._model.to(dtype=torch.float16)
        return self._processor, self._model

    def _detect_all(self, image: Image.Image, query: str) -> list[tuple[tuple[float, ...], float]]:
        import torch

        processor, model = self._load()
        text = query.rstrip(" .") + "."
        inputs = processor(images=image, text=text, return_tensors="pt")
        inputs = {key: value.to(self.config.device) for key, value in inputs.items()}
        with torch.inference_mode():
            outputs = model(**inputs)
        kwargs = {
            "target_sizes": [image.size[::-1]],
            "threshold": self.config.box_threshold,
            "text_threshold": self.config.text_threshold,
        }
        try:
            result = processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                **kwargs,
            )[0]
        except TypeError:
            result = processor.post_process_grounded_object_detection(outputs, **kwargs)[0]
        scores = result["scores"].detach().float().cpu()
        boxes = result["boxes"].detach().float().cpu()
        return [(tuple(float(value) for value in box.tolist()), float(score))
                for box, score in zip(boxes, scores, strict=True)]

    def _detect_one(self, image: Image.Image, query: str) -> tuple[tuple[float, ...], float] | None:
        detections = self._detect_all(image, query)
        return max(detections, key=lambda item: item[1]) if detections else None

    def crop_objects(
        self,
        image_path: Path,
        focus: FocusProgram,
        output_dir: Path,
    ) -> list[ObjectCrop]:
        output_dir.mkdir(parents=True, exist_ok=True)
        with Image.open(image_path) as loaded:
            image = loaded.convert("RGB")
        original_area = image.width * image.height
        crops: list[ObjectCrop] = []
        for index, query in enumerate(focus.grounding_queries, start=1):
            detection = self._detect_one(image, query)
            if detection is None:
                continue
            raw_box, score = detection
            expanded = expand_box(raw_box, image.size, focus.context_margin)
            crop = image.crop(expanded)
            enlarged = enlarge_crop(
                crop,
                self.config.min_short_side,
                max_scale=self.config.max_crop_scale,
            )
            path = output_dir / f"object-crop-{index:02d}.jpg"
            enlarged.save(path, quality=95, optimize=True)
            crops.append(
                ObjectCrop(
                    query=query,
                    score=score,
                    raw_box=raw_box,
                    expanded_box=expanded,
                    path=path.resolve(),
                    area_fraction=(crop.width * crop.height) / original_area,
                )
            )
        return crops


class StaticGrounder:
    """Creates deterministic crops from normalized boxes for tests."""

    def __init__(self, boxes: list[tuple[float, float, float, float]]):
        self.boxes = boxes

    def crop_objects(
        self,
        image_path: Path,
        focus: FocusProgram,
        output_dir: Path,
    ) -> list[ObjectCrop]:
        output_dir.mkdir(parents=True, exist_ok=True)
        with Image.open(image_path) as loaded:
            image = loaded.convert("RGB")
        result: list[ObjectCrop] = []
        for index, (query, normalized) in enumerate(
            zip(focus.grounding_queries, self.boxes, strict=False), start=1
        ):
            x1, y1, x2, y2 = normalized
            raw = (x1 * image.width, y1 * image.height, x2 * image.width, y2 * image.height)
            expanded = expand_box(raw, image.size, focus.context_margin)
            crop = image.crop(expanded)
            path = output_dir / f"object-crop-{index:02d}.jpg"
            crop.save(path, quality=95)
            result.append(
                ObjectCrop(
                    query=query,
                    score=1.0,
                    raw_box=raw,
                    expanded_box=expanded,
                    path=path.resolve(),
                    area_fraction=(crop.width * crop.height) / (image.width * image.height),
                )
            )
        return result
