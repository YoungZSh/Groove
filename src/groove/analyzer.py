"""OpenAI-compatible group-contrastive visual-focus Analyzer."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import time
import urllib.error
import urllib.request
from copy import deepcopy
from dataclasses import dataclass
from difflib import SequenceMatcher
from io import BytesIO
from pathlib import Path
from typing import Protocol

from PIL import Image

from .grounding import expand_box
from .instance_boxes import normalize_instance_boxes, render_instance_boxes
from .schemas import FocusProgram, GroupRollout


SYSTEM_PROMPT = """You are a multimodal visual-evidence Analyzer.

You receive an original image, a question, and successful/failed Student reasoning.
Do not answer the question or reassess the outcome labels. Find the smallest
answer-neutral visual evidence that explains the disagreement.

Use the native tools before returning:
- For text, digits, labels, signs, or documents, use read_text. If the text is tiny
  or its location is uncertain, call ground_image first and pass its bbox to read_text.
- For objects, colors, shapes, counts, spatial relations, materials, or textures,
  use ground_image.

Locate separate targets independently. Inspect every returned visual preview; if it
misses the target, retry with a more precise short English noun phrase.
Every usable tool result has a candidate_id. In the final response, select the best
verified candidate for each required target. Select one candidate for a single target
and multiple candidates only when comparison, counting, or spatial reasoning requires
distinct regions. Never invent an ID or select a crop that misses its target.

Write tool queries and JSON strings in English. visible_focus_instruction must tell
the Teacher what to inspect without revealing an answer or option, rollout outcomes,
rewards, or recognized OCR text.

After tool inspection is complete, return only this JSON object:
{
  "group_summary": "Private summary of the visual disagreement",
  "crucial_evidence": "Smallest sufficient visual evidence",
  "crucial_evidence_type": "text or visual",
  "tool_route": "ocr or dino",
  "visible_focus_instruction": "Short answer-neutral inspection instruction",
  "grounding_queries": ["one to three concrete English visual targets"],
  "selected_candidate_ids": ["one to three candidate IDs from tool results"],
  "confidence": 0.0
}"""

INSTANCE_BOX_GUIDANCE = """

For counting, use ground_instances to inspect candidate objects across the full
image, and use ground_image for ambiguous local details if needed. A ground_instances
candidate is one full-image evidence view containing all returned boxes. Select it
by its candidate_id just like a crop. Verify that the boxes mark distinct relevant
objects; inspect missed objects, duplicates, and false matches. Tool predictions
are not verified counts. Keep the Teacher instruction about visual inspection,
without stating a total or assuming that every box is correct.
"""

COUNTING_GUIDANCE = INSTANCE_BOX_GUIDANCE.replace("ground_instances", "count_objects") + (
    "For an enclosure or spatially restricted counting question, specify the relevant "
    "region from the original image. Check the resulting boxes against that region.\n"
)


class Analyzer(Protocol):
    def analyze(self, group: GroupRollout) -> FocusProgram: ...


def _data_url(path: Path) -> str:
    media_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{payload}"


def _extract_json(text: str) -> dict:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", stripped, re.DOTALL)
    if fenced:
        stripped = fenced.group(1)
    else:
        start, end = stripped.find("{"), stripped.rfind("}")
        if start >= 0 and end > start:
            stripped = stripped[start : end + 1]
    value = json.loads(stripped)
    if not isinstance(value, dict):
        raise ValueError("Analyzer response must be a JSON object")
    return value


def build_group_analysis_text(group: GroupRollout) -> str:
    # Success/failure membership already carries the outcome signal. JSON
    # preserves exact boundaries around untrusted, multiline model outputs;
    # IDs, parsed labels, and numeric rewards add no visual evidence.
    return json.dumps(
        {
            "question": group.question,
            "successful_reasoning": [
                str(item.completion).strip()
                for item in group.rollouts
                if item.is_correct
            ],
            "failed_reasoning": [
                str(item.completion).strip()
                for item in group.rollouts
                if not item.is_correct
            ],
        },
        ensure_ascii=False,
        indent=2,
    )


def _crop_data_url(image_path: Path, bbox: object, max_side: int = 1024) -> str | None:
    """Encode a bounded local crop as a multimodal data URL for the Analyzer."""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        coordinates = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return None
    if not all(value == value and abs(value) != float("inf") for value in coordinates):
        return None
    if max_side <= 0:
        raise ValueError("max_side must be positive")

    with Image.open(image_path) as loaded:
        image = loaded.convert("RGB")
    width, height = image.size
    x1, y1, x2, y2 = coordinates
    x1 = max(0, min(width, round(x1)))
    y1 = max(0, min(height, round(y1)))
    x2 = max(0, min(width, round(x2)))
    y2 = max(0, min(height, round(y2)))
    if x1 >= x2 or y1 >= y2:
        return None
    crop = image.crop((x1, y1, x2, y2))
    scale = min(1.0, max_side / max(crop.size))
    if scale < 1.0:
        crop = crop.resize(
            (max(1, round(crop.width * scale)), max(1, round(crop.height * scale))),
            Image.Resampling.LANCZOS,
        )
    encoded = BytesIO()
    crop.save(encoded, format="JPEG", quality=92, optimize=True)
    payload = base64.b64encode(encoded.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def _tool_feedback_message(
    image_path: Path,
    tool_name: str,
    arguments: dict,
    result: dict,
    max_side: int,
) -> dict | None:
    """Build a user multimodal message containing the exact tool-returned crop."""
    if result.get("error"):
        return None
    if tool_name in {"ground_instances", "count_objects"}:
        if not result.get("found"):
            return None
        with Image.open(image_path) as loaded:
            image = loaded.convert("RGB")
        if result.get("image_size") != list(image.size):
            raise ValueError("instance preview coordinates do not match the original image")
        instances = normalize_instance_boxes(result.get("instances"), image.size)
        preview = render_instance_boxes(image, instances)
        if max_side <= 0:
            raise ValueError("max_side must be positive")
        preview.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        encoded = BytesIO()
        preview.save(encoded, format="PNG")
        url = "data:image/png;base64," + base64.b64encode(encoded.getvalue()).decode("ascii")
        return {"role": "user", "content": [
            {"type": "text", "text": (
                f"Candidate `{result.get('candidate_id', '')}`. This is the original image "
                f"with candidate instance boxes for `{arguments.get('query') or arguments.get('target', '')}`. "
                "Inspect each box against the original image for false matches, duplicate "
                "coverage, and missed objects. Select this candidate only if its visual "
                "evidence is useful; the boxes do not constitute a verified count."
            )},
            {"type": "image_url", "image_url": {"url": url}},
        ]}
    bbox_key = "bbox" if tool_name == "ground_image" else "crop_bbox"
    bbox = result.get(bbox_key)
    preview_url = _crop_data_url(image_path, bbox, max_side=max_side)
    if preview_url is None:
        return None
    query = str(arguments.get("query") or result.get("query") or tool_name).strip()
    candidate_id = str(result.get("candidate_id", "")).strip()
    candidate_text = f"Candidate `{candidate_id}`. " if candidate_id else ""
    score = result.get("score")
    score_text = f" with score {float(score):.3f}" if isinstance(score, (int, float)) else ""
    coordinates = ", ".join(str(value) for value in bbox)
    return {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    f"{candidate_text}Visual feedback for the `{tool_name}` query `{query}`{score_text}. "
                    f"The attached image is the exact local crop returned by the tool "
                    f"(original-image bbox: [{coordinates}]). Inspect this crop against "
                    "the original image. If it does not contain the requested target, "
                    "rewrite the query as a short English noun phrase and call the tool again."
                ),
            },
            {"type": "image_url", "image_url": {"url": preview_url}},
        ],
    }


def _tight_ocr_text_bbox(
    result: dict,
    reference_text: str = "",
) -> tuple[float, float, float, float] | None:
    """Return the tight union of meaningful OCR line polygons.

    ``crop_bbox`` is deliberately allowed to be a large search context (the OCR
    worker expands narrow detector boxes before scanning).  It must not be used
    as the Teacher evidence crop when OCR returned text polygons.  Ignore
    punctuation-only detector noise such as isolated dashes when possible.
    """
    lines = result.get("text") if isinstance(result, dict) else None
    if not isinstance(lines, list):
        return None

    candidates: list[tuple[dict, tuple[float, float, float, float], float]] = []
    fallback: list[tuple[dict, tuple[float, float, float, float], float]] = []
    for line in lines:
        if not isinstance(line, dict) or not isinstance(line.get("bbox"), list):
            continue
        points = line["bbox"]
        if not points:
            continue
        valid = True
        for point in points:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                valid = False
                break
            try:
                float(point[0])
                float(point[1])
            except (TypeError, ValueError):
                valid = False
                break
        if not valid:
            continue
        coordinates = [
            (float(point[0]), float(point[1]))
            for point in points
        ]
        line_box = (
            min(point[0] for point in coordinates),
            min(point[1] for point in coordinates),
            max(point[0] for point in coordinates),
            max(point[1] for point in coordinates),
        )
        try:
            confidence = float(line.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        entry = (line, line_box, confidence)
        fallback.append(entry)
        if re.search(r"[A-Za-z0-9]", str(line.get("text", ""))):
            candidates.append(entry)

    selected = candidates or fallback
    if not selected:
        return None
    if len(selected) > 1:
        # OCR search contexts can contain unrelated text fragments.  Keep the
        # line that best matches the Analyzer's private question/evidence text,
        # falling back to confidence when no textual match is available.  This
        # avoids selecting a high-confidence operator logo over a lower-
        # confidence answer-bearing label (for example, ``DHPENEWS`` versus
        # ``TechnipFMC`` on the vessel image).
        reference_terms = []
        option_texts = re.findall(r"(?m)^\s*[A-D][.)]\s*(.+?)\s*$", reference_text)
        reference_source = " ".join(option_texts) if option_texts else reference_text
        reference_words = re.findall(r"[a-z0-9]+", reference_source.lower())
        reference_terms.extend(word for word in reference_words if len(word) >= 3)
        for ngram_size in (2, 3, 4):
            reference_terms.extend(
                "".join(reference_words[index : index + ngram_size])
                for index in range(len(reference_words) - ngram_size + 1)
            )
        reference_terms = list(dict.fromkeys(reference_terms))

        def selection_key(entry):
            line, _box, confidence = entry
            normalized_line = re.sub(r"[^a-z0-9]+", "", str(line.get("text", "")).lower())
            similarity = max(
                (
                    SequenceMatcher(None, normalized_line, term).ratio()
                    for term in reference_terms
                    if normalized_line and term
                ),
                default=0.0,
            )
            if similarity < 0.45:
                similarity = 0.0
            return (0.7 * similarity + 0.3 * confidence, confidence)

        best = max(selected, key=selection_key)
        _, best_box, _best_confidence = best
        best_height = max(1.0, best_box[3] - best_box[1])
        nearby = []
        for entry in selected:
            _, box, _confidence = entry
            vertical_gap = max(0.0, max(best_box[1], box[1]) - min(best_box[3], box[3]))
            horizontal_gap = max(0.0, max(best_box[0], box[0]) - min(best_box[2], box[2]))
            if vertical_gap <= max(2.5 * best_height, 24.0) and horizontal_gap <= max(6.0 * best_height, 48.0):
                nearby.append(entry)
        selected = nearby or [best]
    coordinates = [point for _, box, _ in selected for point in ((box[0], box[1]), (box[2], box[3]))]
    x_values = [point[0] for point in coordinates]
    y_values = [point[1] for point in coordinates]
    return min(x_values), min(y_values), max(x_values), max(y_values)


def _ocr_confidence_for_bbox(result: dict, bbox: tuple[float, float, float, float]) -> float:
    """Return the confidence of OCR lines covered by a selected text box."""
    lines = result.get("text") if isinstance(result, dict) else None
    if not isinstance(lines, list):
        return 0.0
    x1, y1, x2, y2 = bbox
    confidences = []
    for line in lines:
        if not isinstance(line, dict) or not isinstance(line.get("bbox"), list):
            continue
        points = line["bbox"]
        try:
            line_x1 = min(float(point[0]) for point in points)
            line_y1 = min(float(point[1]) for point in points)
            line_x2 = max(float(point[0]) for point in points)
            line_y2 = max(float(point[1]) for point in points)
            confidence = float(line.get("confidence", 0.0))
        except (TypeError, ValueError, IndexError):
            continue
        if line_x2 >= x1 and line_x1 <= x2 and line_y2 >= y1 and line_y1 <= y2:
            confidences.append(confidence)
    return max(confidences, default=0.0)


@dataclass(frozen=True)
class OpenAIAnalyzerConfig:
    base_url: str
    api_key: str
    model: str = "gpt-5.6"
    timeout_seconds: float = 180.0
    temperature: float = 0.0
    max_completion_tokens: int = 512
    max_retries: int = 5
    retry_delay_seconds: float = 1.0
    tool_feedback_max_side: int = 1024
    disable_thinking: bool = False
    use_vision_tools: bool = False
    max_tool_rounds: int = 3

    @classmethod
    def from_env(cls) -> "OpenAIAnalyzerConfig":
        base_url = os.environ.get("ANALYZER_BASE_URL", "").strip()
        api_key = os.environ.get("ANALYZER_API_KEY", "").strip()
        model = os.environ.get("ANALYZER_MODEL", "gpt-5.6").strip()
        if not base_url or not api_key:
            raise RuntimeError("ANALYZER_BASE_URL and ANALYZER_API_KEY are required")
        return cls(
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout_seconds=float(os.environ.get("ANALYZER_TIMEOUT_SECONDS", "180")),
            temperature=float(os.environ.get("ANALYZER_TEMPERATURE", "0.0")),
            max_completion_tokens=int(
                os.environ.get("ANALYZER_MAX_COMPLETION_TOKENS", "512")
            ),
            max_retries=int(os.environ.get("ANALYZER_API_RETRIES", "5")),
            retry_delay_seconds=float(os.environ.get("ANALYZER_API_RETRY_DELAY", "1.0")),
            tool_feedback_max_side=int(
                os.environ.get("ANALYZER_TOOL_FEEDBACK_MAX_SIDE", "1024")
            ),
            disable_thinking=os.environ.get("ANALYZER_DISABLE_THINKING", "false")
            .strip()
            .lower()
            in {"1", "true", "yes", "on"},
            use_vision_tools=os.environ.get("ANALYZER_USE_VISION_TOOLS", "false")
            .strip()
            .lower()
            in {"1", "true", "yes", "on"},
            max_tool_rounds=int(os.environ.get("ANALYZER_MAX_TOOL_ROUNDS", "3")),
        )


class OpenAICompatibleAnalyzer:
    def __init__(self, config: OpenAIAnalyzerConfig):
        self.config = config
        # Audit-only trace for probe runs.  It is never sent to the Student or
        # injected into a Teacher prompt.
        self.last_tool_trace: list[dict] = []

    def _endpoint(self) -> str:
        base = self.config.base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        if not base.endswith("/v1"):
            base += "/v1"
        return base + "/chat/completions"

    def _request(self, body: dict) -> dict:
        request = urllib.request.Request(
            self._endpoint(),
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        payload = None
        for retry_index in range(self.config.max_retries + 1):
            try:
                with urllib.request.urlopen(
                    request, timeout=self.config.timeout_seconds
                ) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                retryable = exc.code == 429 or exc.code >= 500
                if not retryable or retry_index >= self.config.max_retries:
                    raise RuntimeError(f"Analyzer HTTP {exc.code}: {detail[:1000]}") from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                if retry_index >= self.config.max_retries:
                    raise RuntimeError(f"Analyzer request failed: {exc}") from exc
            time.sleep(self.config.retry_delay_seconds)
        if payload is None:
            raise RuntimeError("Analyzer request exhausted retries without a response")
        return payload

    @staticmethod
    def _message_content(message: dict) -> str:
        content_text = message.get("content") or ""
        if isinstance(content_text, list):
            content_text = "".join(
                item.get("text", "") for item in content_text if isinstance(item, dict)
            )
        return str(content_text)

    def _base_body(self, messages: list[dict]) -> dict:
        body = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_completion_tokens": self.config.max_completion_tokens,
        }
        # Qwen3.5's vLLM chat template supports this flag.  Keep it opt-in so
        # generic OpenAI-compatible endpoints that reject template kwargs remain
        # usable.
        if self.config.disable_thinking:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        return body

    def _analyze_with_tools(self, group: GroupRollout, messages: list[dict]) -> FocusProgram:
        """Run a bounded Analyzer-only function-call loop.

        The tool registry is deliberately not connected to VERL's student agent
        loop.  It only lets the external Qwen Analyzer inspect evidence before
        producing the private focus program for the Teacher branch.
        """
        from .analyzer_tools import AnalyzerVisionToolRegistry

        registry = AnalyzerVisionToolRegistry()
        if self.config.tool_feedback_max_side <= 0:
            raise ValueError("ANALYZER_TOOL_FEEDBACK_MAX_SIDE must be positive")
        tool_messages = deepcopy(messages)
        tool_names = {schema.get("function", {}).get("name") for schema in registry.schemas}
        if "count_objects" in tool_names:
            tool_messages[0]["content"] += COUNTING_GUIDANCE
        elif "ground_instances" in tool_names:
            tool_messages[0]["content"] += INSTANCE_BOX_GUIDANCE
        self.last_tool_trace = []
        for _tool_round in range(self.config.max_tool_rounds):
            body = self._base_body(tool_messages)
            body.update({"tools": registry.schemas, "tool_choice": "auto"})
            message = self._request(body)["choices"][0]["message"]
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                return self._finalize_focus_selection(group, message, tool_messages)

            tool_messages.append(
                {
                    "role": "assistant",
                    "content": self._message_content(message),
                    "tool_calls": tool_calls,
                }
            )
            for call in tool_calls:
                call_id = str(call.get("id", ""))
                candidate_id = f"candidate_{len(self.last_tool_trace) + 1}"
                function = call.get("function", {})
                name = str(function.get("name", ""))
                raw_arguments = function.get("arguments", "{}")
                arguments: dict = {}
                try:
                    arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
                    if not isinstance(arguments, dict):
                        raise ValueError("tool arguments must be a JSON object")
                    result = registry.execute(group.image_path, name, arguments)
                except Exception as exc:
                    result = {"error": f"{type(exc).__name__}: {exc}"}
                if not isinstance(result, dict):
                    result = {"error": f"Tool returned {type(result).__name__}, expected a dictionary"}
                result = {**result, "candidate_id": candidate_id}
                self.last_tool_trace.append(
                    {
                        "round": _tool_round + 1,
                        "candidate_id": candidate_id,
                        "name": name,
                        "arguments": arguments,
                        "result": result,
                    }
                )
                tool_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )
                feedback = _tool_feedback_message(
                    group.image_path,
                    name,
                    arguments,
                    result,
                    max_side=self.config.tool_feedback_max_side,
                )
                if feedback is not None:
                    tool_messages.append(feedback)
                    self.last_tool_trace[-1]["visual_feedback_attached"] = True
        # Some Qwen versions keep proposing another inspection even after the
        # requested evidence has been returned.  End the tool phase explicitly
        # and obtain the required structured focus program without exposing any
        # tool machinery to the student.
        tool_messages.append(
            {
                "role": "user",
                "content": (
                    "The visual-tool call budget is exhausted. Use the original image and "
                    "the returned visual previews to finish verification without calling "
                    "another tool, then return exactly the required JSON object."
                ),
            }
        )
        message = self._request(self._base_body(tool_messages))["choices"][0]["message"]
        return self._finalize_focus_selection(group, message, tool_messages)

    def _candidate_selection_summary(self) -> list[dict]:
        candidates = []
        for trace in self.last_tool_trace:
            result = trace.get("result", {})
            if not isinstance(result, dict) or result.get("error"):
                continue
            name = str(trace.get("name", ""))
            if name in {"ground_instances", "count_objects"}:
                size = result.get("image_size")
                if not result.get("found") or not isinstance(size, list) or len(size) != 2:
                    continue
                instances = normalize_instance_boxes(result.get("instances"), tuple(size))
                if instances:
                    candidates.append({
                        "candidate_id": trace.get("candidate_id"), "tool": name,
                        "query": trace.get("arguments", {}).get("query", result.get("query", "")),
                        "bbox": [0, 0, *size], "kind": "instance_boxes",
                        "instance_candidates": len(instances),
                    })
                continue
            bbox_key = "bbox" if name == "ground_image" else "crop_bbox"
            bbox = result.get(bbox_key)
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            if name == "ground_image" and not result.get("found", True):
                continue
            candidates.append(
                {
                    "candidate_id": trace.get("candidate_id"),
                    "tool": name,
                    "query": trace.get("arguments", {}).get("query", result.get("query", "")),
                    "bbox": bbox,
                }
            )
        return candidates

    def _finalize_focus_selection(
        self,
        group: GroupRollout,
        message: dict,
        tool_messages: list[dict],
    ) -> FocusProgram:
        content = self._message_content(message)
        try:
            focus = FocusProgram.model_validate(_extract_json(content))
            return self._attach_tool_regions(
                focus,
                group.image_path,
                self._ocr_reference_text(group, message),
            )
        except (TypeError, ValueError) as first_error:
            candidates = self._candidate_selection_summary()
            if not candidates:
                raise ValueError("Analyzer produced no selectable evidence candidates") from first_error
            repair_messages = list(tool_messages)
            repair_messages.append({"role": "assistant", "content": content})
            repair_messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your final candidate selection was missing or invalid. Do not call another tool. "
                        "Return the required JSON object again, selecting one to three IDs only from these "
                        f"verified candidates:\n{json.dumps(candidates, ensure_ascii=False, indent=2)}"
                    ),
                }
            )
            repaired_message = self._request(self._base_body(repair_messages))["choices"][0]["message"]
            repaired_focus = FocusProgram.model_validate(
                _extract_json(self._message_content(repaired_message))
            )
            return self._attach_tool_regions(
                repaired_focus,
                group.image_path,
                self._ocr_reference_text(group, repaired_message),
            )

    @staticmethod
    def _ocr_reference_text(group: GroupRollout, message: dict) -> str:
        """Build private text anchors for selecting among OCR candidates."""
        parts = [group.question, str(message.get("content", ""))]
        return " ".join(parts)

    def _attach_tool_regions(
        self,
        focus: FocusProgram,
        image_path: Path,
        reference_text: str = "",
    ) -> FocusProgram:
        """Promote exactly the candidates selected after visual verification."""
        from .schemas import ToolRegion

        candidate_regions: dict[str, ToolRegion] = {}
        with Image.open(image_path) as loaded:
            image_size = loaded.size
        for trace in self.last_tool_trace:
            result = trace.get("result", {})
            if not isinstance(result, dict) or result.get("error"):
                continue
            candidate_id = str(trace.get("candidate_id") or result.get("candidate_id") or "").strip()
            if not candidate_id:
                continue
            if candidate_id in candidate_regions:
                raise ValueError(f"Duplicate Analyzer candidate ID: {candidate_id}")
            if trace.get("name") in {"ground_instances", "count_objects"} and result.get("found"):
                if result.get("image_size") != list(image_size):
                    raise ValueError("instance evidence coordinates do not match the original image")
                instances = normalize_instance_boxes(result.get("instances"), image_size)
                if instances:
                    candidate_regions[candidate_id] = ToolRegion(
                        query=str(result.get("query", "object instances")),
                        expanded_box=(0, 0, *image_size),
                        score=sum(item.score for item in instances) / len(instances),
                        source="grounding_dino", kind="instance_boxes", instances=instances,
                    )
            elif trace.get("name") == "read_text" and isinstance(result.get("crop_bbox"), list):
                bbox = result["crop_bbox"]
                if len(bbox) == 4:
                    tight_bbox = _tight_ocr_text_bbox(result, reference_text=reference_text)
                    if tight_bbox is not None:
                        # OCR's crop_bbox is a search context.  The Teacher
                        # should receive the recognized text itself plus the
                        # same 10–15% context margin used by DINO crops.
                        bbox = list(expand_box(tight_bbox, image_size, focus.context_margin))
                        selected_confidence = _ocr_confidence_for_bbox(result, tight_bbox)
                    else:
                        selected_confidence = 0.0
                    confidences = [
                        float(item.get("confidence", 0.0))
                        for item in result.get("text", [])
                        if isinstance(item, dict)
                    ]
                    query = "OCR search context"
                    if result.get("text"):
                        query = "OCR context: " + ", ".join(
                            str(item.get("text", "")) for item in result["text"][:3] if isinstance(item, dict)
                        )
                    candidate_regions[candidate_id] = ToolRegion(
                        query=query,
                        expanded_box=tuple(int(value) for value in bbox),
                        score=(selected_confidence if tight_bbox is not None else max(confidences, default=0.0)),
                        source="paddle_ocr_text" if tight_bbox is not None else "paddle_ocr_context",
                    )
            elif (
                trace.get("name") == "ground_image"
                and result.get("found", True)
                and isinstance(result.get("bbox"), list)
            ):
                bbox = result["bbox"]
                if len(bbox) == 4:
                    candidate_regions[candidate_id] = ToolRegion(
                        query=str(result.get("query", "Grounding DINO target")),
                        expanded_box=tuple(int(value) for value in bbox),
                        score=float(result.get("score", 0.0)),
                        source="grounding_dino",
                    )

        selected_ids = focus.selected_candidate_ids
        if not selected_ids:
            raise ValueError("Analyzer must select at least one evidence candidate")
        missing = [candidate_id for candidate_id in selected_ids if candidate_id not in candidate_regions]
        if missing:
            raise ValueError(f"Analyzer selected unavailable candidate IDs: {missing}")
        selected_regions = [candidate_regions[candidate_id] for candidate_id in selected_ids]
        return focus.model_copy(update={"tool_regions": selected_regions})

    def analyze(self, group: GroupRollout) -> FocusProgram:
        content = [
            {"type": "image_url", "image_url": {"url": _data_url(group.image_path)}},
            {"type": "text", "text": build_group_analysis_text(group)},
        ]
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
        ]
        messages.append({"role": "user", "content": content})
        if self.config.use_vision_tools:
            return self._analyze_with_tools(group, messages)
        payload = self._request(self._base_body(messages))
        return FocusProgram.model_validate(_extract_json(self._message_content(payload["choices"][0]["message"])))


class StaticAnalyzer:
    """Deterministic Analyzer used by unit and pipeline smoke tests."""

    def __init__(self, focus: FocusProgram):
        self.focus = focus

    def analyze(self, group: GroupRollout) -> FocusProgram:
        del group
        return self.focus
