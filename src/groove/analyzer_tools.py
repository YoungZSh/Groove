"""Local visual tools available only to the external Analyzer agent.

The student policy remains a tool-free image-to-text model.  These wrappers are
called by the Qwen Analyzer while constructing privileged Teacher evidence.
"""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from PIL import Image

from .grounding import GroundingDinoConfig, GroundingDinoGrounder, expand_box
from .counting import DEFAULT_COUNTING_PROFILE, counting_region, select_counting_instances
from .instance_boxes import normalize_instance_boxes
from .schemas import InstanceBox


NON_ENGLISH_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


GROUND_IMAGE_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "ground_image",
        "description": (
            "Use Grounding DINO to localize one concrete object or region in the current image. "
            "The query must be a short, concrete English noun phrase; for tiny text, first "
            "localize its text carrier and then pass the bbox to OCR."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A concrete English visual target, for example \"small street sign under the bridge\".",
                },
                "context_margin": {
                    "type": "number",
                    "description": "Context expansion ratio. The program clamps it to 0.10–0.15; default 0.12.",
                },
            },
            "required": ["query"],
        },
    },
}


READ_TEXT_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "read_text",
        "description": (
            "Use GPU PP-OCR to read scene text. Prefer the bbox returned by ground_image; "
            "the tool enlarges that region and returns recognized text with confidence."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "bbox": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 4,
                    "maxItems": 4,
                    "description": (
                        "Optional [x1, y1, x2, y2] in original-image pixels. For a small DINO box, "
                        "the tool automatically searches surrounding sign context before OCR."
                    ),
                },
                "scale": {
                    "type": "integer",
                    "description": "Optional crop enlargement from 1 to 8; default 4.",
                },
            },
            "required": [],
        },
    },
}


ANALYZER_TOOL_SCHEMAS = [GROUND_IMAGE_TOOL_SCHEMA, READ_TEXT_TOOL_SCHEMA]

GROUND_INSTANCES_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "ground_instances",
        "description": (
            "Find candidate instances of one object category throughout the original image. "
            "Use for counting: a full-image preview shows an unfilled box around each candidate. "
            "Check for missed objects, duplicate boxes, and false matches before selecting the evidence. "
            "The returned candidates are predictions, not a verified count."
        ),
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "A short English object category, such as sheep."}},
            "required": ["query"],
        },
    },
}

COUNT_OBJECTS_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "count_objects",
        "description": (
            "Detect and count candidate instances of one object category and return an original-image "
            "preview with boxes. Filtering is fixed inside this tool. For a question restricted to "
            "an enclosure or image area, provide that region in original-image pixels. Verify missed "
            "objects and false matches; the estimated count is not a ground-truth answer."
        ),
        "parameters": {"type": "object", "properties": {
            "target": {"type": "string", "description": "Short English object category, for example sheep."},
            "region": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4,
                       "description": "Optional [x1, y1, x2, y2] limiting counting to the relevant scene region."},
        }, "required": ["target"]},
    },
}


@dataclass(frozen=True)
class AnalyzerVisionToolConfig:
    ocr_python: Path = Path("/home/yzs/miniconda3/envs/ocr-paddle-gpu/bin/python")
    ocr_script: Path = Path(__file__).resolve().parents[2] / "scripts" / "paddle_ocr_tool.py"
    ocr_gpu_id: str = "1"
    ocr_timeout_seconds: float = 120.0
    grounding_device: str = "cpu"
    grounding_box_threshold: float = 0.15
    grounding_text_threshold: float = 0.15
    grounding_model: str = "IDEA-Research/grounding-dino-base"
    grounding_local_files_only: bool = True
    grounding_url: str = ""
    ocr_url: str = ""
    enable_instance_boxes: bool = False
    enable_counting: bool = False

    @classmethod
    def from_env(cls) -> "AnalyzerVisionToolConfig":
        return cls(
            ocr_python=Path(
                os.environ.get(
                    "ANALYZER_OCR_PYTHON", "/home/yzs/miniconda3/envs/ocr-paddle-gpu/bin/python"
                )
            ),
            ocr_script=Path(
                os.environ.get(
                    "ANALYZER_OCR_SCRIPT",
                    str(Path(__file__).resolve().parents[2] / "scripts" / "paddle_ocr_tool.py"),
                )
            ),
            ocr_gpu_id=os.environ.get("ANALYZER_OCR_GPU_ID", "1"),
            ocr_timeout_seconds=float(os.environ.get("ANALYZER_OCR_TIMEOUT_SECONDS", "120")),
            grounding_device=os.environ.get("ANALYZER_GROUNDING_DEVICE", "cpu"),
            grounding_box_threshold=float(os.environ.get("ANALYZER_GROUNDING_BOX_THRESHOLD", "0.15")),
            grounding_text_threshold=float(os.environ.get("ANALYZER_GROUNDING_TEXT_THRESHOLD", "0.15")),
            grounding_model=os.environ.get(
                "ANALYZER_GROUNDING_MODEL", "IDEA-Research/grounding-dino-base"
            ),
            grounding_local_files_only=os.environ.get(
                "ANALYZER_GROUNDING_LOCAL_FILES_ONLY", "true"
            ).lower()
            in {"1", "true", "yes", "on"},
            grounding_url=os.environ.get("ANALYZER_GROUNDING_URL", "").rstrip("/"),
            ocr_url=os.environ.get("ANALYZER_OCR_URL", "").rstrip("/"),
            enable_instance_boxes=os.environ.get("ANALYZER_ENABLE_INSTANCE_BOXES", "false").lower()
            in {"1", "true", "yes", "on"},
            enable_counting=os.environ.get("ANALYZER_ENABLE_COUNTING", "false").lower()
            in {"1", "true", "yes", "on"},
        )


class AnalyzerVisionToolRegistry:
    """Small, explicit local registry for the Qwen Analyzer's two visual tools."""

    def __init__(self, config: AnalyzerVisionToolConfig | None = None):
        self.config = config or AnalyzerVisionToolConfig.from_env()
        if not self.config.ocr_url and not self.config.ocr_python.is_file():
            raise FileNotFoundError(f"Analyzer OCR Python is unavailable: {self.config.ocr_python}")
        if not self.config.ocr_url and not self.config.ocr_script.is_file():
            raise FileNotFoundError(f"Analyzer OCR runner is unavailable: {self.config.ocr_script}")
        self._grounder = None
        self._counting_grounder = None
        if not self.config.grounding_url:
            self._grounder = GroundingDinoGrounder(
                GroundingDinoConfig(
                    model=self.config.grounding_model,
                    device=self.config.grounding_device,
                    box_threshold=self.config.grounding_box_threshold,
                    text_threshold=self.config.grounding_text_threshold,
                    local_files_only=self.config.grounding_local_files_only,
                )
            )

    @staticmethod
    def _remote_call(url: str, image_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
        body = dict(payload)
        body["image_base64"] = base64.b64encode(image_path.read_bytes()).decode("ascii")
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        retries = int(os.environ.get("ANALYZER_TOOL_API_RETRIES", "3"))
        retry_delay = float(os.environ.get("ANALYZER_TOOL_API_RETRY_DELAY", "0.25"))
        if retries < 0:
            raise ValueError("ANALYZER_TOOL_API_RETRIES must be non-negative")
        if retry_delay < 0:
            raise ValueError("ANALYZER_TOOL_API_RETRY_DELAY must be non-negative")

        for attempt in range(retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=180) as response:
                    result = json.loads(response.read().decode("utf-8"))
                if not isinstance(result, dict) or "error" in result:
                    raise RuntimeError(f"remote Analyzer tool failed: {result}")
                return result
            except urllib.error.HTTPError as exc:
                # Input/model errors are deterministic and should be surfaced
                # immediately; only transient server errors are retried.
                if exc.code < 500 or attempt >= retries:
                    raise
            except (urllib.error.URLError, TimeoutError):
                if attempt >= retries:
                    raise
            time.sleep(retry_delay * (attempt + 1))

        raise RuntimeError("remote Analyzer tool request exhausted retries")

    @property
    def schemas(self) -> list[dict[str, Any]]:
        if self.config.enable_counting:
            return [*ANALYZER_TOOL_SCHEMAS, COUNT_OBJECTS_TOOL_SCHEMA]
        return [*ANALYZER_TOOL_SCHEMAS, GROUND_INSTANCES_TOOL_SCHEMA] if self.config.enable_instance_boxes else ANALYZER_TOOL_SCHEMAS

    @staticmethod
    def _bbox(value: Any, image_size: tuple[int, int]) -> tuple[float, float, float, float]:
        if not isinstance(value, list) or len(value) != 4:
            raise ValueError("bbox must be [x1, y1, x2, y2]")
        x1, y1, x2, y2 = (float(item) for item in value)
        width, height = image_size
        x1, x2 = max(0.0, x1), min(float(width), x2)
        y1, y2 = max(0.0, y1), min(float(height), y2)
        if x1 >= x2 or y1 >= y2:
            raise ValueError("bbox must lie within the image and have positive area")
        return x1, y1, x2, y2

    def ground_image(self, image_path: Path, query: str, context_margin: float = 0.12) -> dict[str, Any]:
        query = str(query).strip()
        if not query:
            raise ValueError("ground_image requires a non-empty query")
        if NON_ENGLISH_CJK.search(query):
            raise ValueError(
                "ground_image requires an English query; rewrite the target as a short English noun phrase"
            )
        requested_margin = float(context_margin)
        margin = min(0.15, max(0.10, requested_margin))
        if self.config.grounding_url:
            return self._remote_call(
                self.config.grounding_url,
                image_path,
                {
                    "query": query,
                    "context_margin": margin,
                    "box_threshold": self.config.grounding_box_threshold,
                    "text_threshold": self.config.grounding_text_threshold,
                },
            )
        with Image.open(image_path) as loaded:
            image = loaded.convert("RGB")
        assert self._grounder is not None
        result = self._grounder._detect_one(image, query)
        if result is None:
            return {"query": query, "found": False, "image_size": list(image.size)}
        raw_box, score = result
        expanded = expand_box(raw_box, image.size, margin)
        return {
                "query": query,
                "found": True,
                "score": score,
                "requested_context_margin": requested_margin,
                "applied_context_margin": margin,
                "raw_bbox": [round(value, 2) for value in raw_box],
            "bbox": list(expanded),
            "image_size": list(image.size),
        }

    def ground_instances(self, image_path: Path, query: str) -> dict[str, Any]:
        if not self.config.enable_instance_boxes:
            raise ValueError("ground_instances is not enabled")
        query = str(query).strip()
        if not query or NON_ENGLISH_CJK.search(query):
            raise ValueError("ground_instances requires a nonempty English object query")
        with Image.open(image_path) as loaded:
            image_size = loaded.size
        if self.config.grounding_url:
            result = self._remote_call(self.config.grounding_url, image_path, {
                "query": query, "return_all": True,
                "box_threshold": self.config.grounding_box_threshold,
                "text_threshold": self.config.grounding_text_threshold,
            })
            if "instances" not in result:
                raise ValueError("DINO endpoint lacks multi-instance support; update its server before enabling ground_instances")
            if result.get("image_size") != list(image_size):
                raise ValueError("DINO instance coordinates do not match the original image size")
            values = result["instances"]
        else:
            with Image.open(image_path) as loaded:
                image = loaded.convert("RGB")
            assert self._grounder is not None
            values = [{"bbox": list(box), "score": score} for box, score in self._grounder._detect_all(image, query)]
        instances = normalize_instance_boxes(values, image_size)
        return {"query": query, "found": bool(instances), "image_size": list(image_size),
                "instances": [item.model_dump(mode="json") for item in instances]}

    def count_objects(self, image_path: Path, target: str, region=None) -> dict[str, Any]:
        if not self.config.enable_counting:
            raise ValueError("count_objects is not enabled")
        target = str(target).strip()
        if not target or NON_ENGLISH_CJK.search(target):
            raise ValueError("count_objects requires a nonempty English target")
        with Image.open(image_path) as loaded:
            image = loaded.convert("RGB")
        roi = counting_region(region, image.size)
        cropped = image.crop(roi)
        profile = DEFAULT_COUNTING_PROFILE
        if self.config.grounding_url:
            # The remote detector sees the ROI itself. Its boxes must be mapped
            # back to the original coordinate system before drawing evidence.
            with TemporaryDirectory(prefix="groove-counting-") as temporary:
                path = Path(temporary) / "region.png"
                cropped.save(path)
                result = self._remote_call(self.config.grounding_url, path, {
                    "query": target, "return_all": True,
                    "box_threshold": profile.box_threshold, "text_threshold": profile.text_threshold,
                })
            if "instances" not in result:
                raise ValueError("DINO endpoint lacks multi-instance support for count_objects")
            if result.get("image_size") != list(cropped.size):
                raise ValueError("counting detector coordinates do not match the requested region")
            raw = result["instances"]
        else:
            if self._counting_grounder is None:
                self._counting_grounder = GroundingDinoGrounder(GroundingDinoConfig(
                    model=self.config.grounding_model, device=self.config.grounding_device,
                    local_files_only=self.config.grounding_local_files_only,
                    box_threshold=profile.box_threshold, text_threshold=profile.text_threshold))
            raw = [{"bbox": list(box), "score": score}
                   for box, score in self._counting_grounder._detect_all(cropped, target)]
        candidates = normalize_instance_boxes(raw, cropped.size)
        indices = select_counting_instances(candidates, profile)
        original_candidates = [InstanceBox(bbox=(item.bbox[0]+roi[0], item.bbox[1]+roi[1],
                                                item.bbox[2]+roi[0], item.bbox[3]+roi[1]), score=item.score)
                               for item in candidates]
        instances = [original_candidates[i] for i in indices]
        return {"query": target, "found": bool(instances), "image_size": list(image.size),
                "region": list(roi), "estimated_count": len(instances), "backend": "grounding_dino",
                "raw_candidate_count": len(candidates), "kept_candidate_indices": indices,
                "candidates_before_filtering": [item.model_dump(mode="json") for item in original_candidates],
                "profile": {"box_threshold": profile.box_threshold, "text_threshold": profile.text_threshold,
                            "nms_iou_threshold": profile.nms_iou_threshold},
                "instances": [item.model_dump(mode="json") for item in instances]}

    def read_text(
        self,
        image_path: Path,
        bbox: Any = None,
        scale: int = 4,
    ) -> dict[str, Any]:
        with Image.open(image_path) as loaded:
            image_size = loaded.size
        normalized_bbox = self._bbox(bbox, image_size) if bbox is not None else None
        # A detector box around one plate in a signpost commonly excludes the
        # target street-name plate immediately above or below it.  For narrow
        # text boxes, enlarge the *OCR search* crop; this is independent of
        # teacher-image cropping and avoids upscaling only a wrong sign.
        ocr_bbox = normalized_bbox
        if normalized_bbox is not None:
            width, height = image_size
            area_fraction = ((normalized_bbox[2] - normalized_bbox[0]) * (normalized_bbox[3] - normalized_bbox[1])) / (
                width * height
            )
            if area_fraction < 0.1:
                ocr_bbox = expand_box(normalized_bbox, image_size, margin=2.0)
        if self.config.ocr_url:
            return self._remote_call(
                self.config.ocr_url,
                image_path,
                {"bbox": list(ocr_bbox) if ocr_bbox is not None else None, "scale": int(scale)},
            )
        command = [str(self.config.ocr_python), str(self.config.ocr_script), "--image", str(image_path)]
        if ocr_bbox is not None:
            command.extend(["--bbox", ",".join(str(round(item, 2)) for item in ocr_bbox)])
        command.extend(["--scale", str(int(scale))])
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self.config.ocr_gpu_id
        completed = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            timeout=self.config.ocr_timeout_seconds,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "PaddleOCR failed: " + (completed.stderr or completed.stdout)[-1500:]
            )
        for line in reversed(completed.stdout.splitlines()):
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and "text" in value:
                return value
        raise RuntimeError("PaddleOCR returned no JSON tool result")

    def execute(self, image_path: Path, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "count_objects":
            if set(arguments) - {"target", "region"}:
                raise ValueError("count_objects accepts only target and region; filtering is fixed")
            return self.count_objects(image_path, target=arguments.get("target", ""), region=arguments.get("region"))
        if name == "ground_instances":
            return self.ground_instances(image_path, query=arguments.get("query", ""))
        if name == "ground_image":
            return self.ground_image(
                image_path,
                query=arguments.get("query", ""),
                context_margin=arguments.get("context_margin", 0.12),
            )
        if name == "read_text":
            return self.read_text(
                image_path,
                bbox=arguments.get("bbox"),
                scale=arguments.get("scale", 4),
            )
        raise ValueError(f"Unknown Analyzer visual tool: {name}")
