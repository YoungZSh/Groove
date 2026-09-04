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
from dataclasses import dataclass
from difflib import SequenceMatcher
from io import BytesIO
from pathlib import Path
from typing import Protocol

from PIL import Image

from .grounding import expand_box
from .schemas import FocusProgram, GroupRollout


SYSTEM_PROMPT = """You are the multimodal self-evolution Analyzer.

You receive the original image, the question, and trajectories that the program has
already divided into successful and failed Student rollouts. Do not decide the reward
and do not answer the question directly.

Compare the two rollout groups and find the smallest sufficient Crucial Evidence: an
observable visual detail that explains their disagreement and could correct the key
visual mistake. It may be text, an object, a local attribute, a count, a relation, a
mark, or a texture. It must not be the answer itself or a restatement of a successful
rollout.

Follow this procedure:
1. Compare what the successful and failed rollouts relied on, and identify the key
   visual disagreement.
2. Summarize the Crucial Evidence in one sentence.
3. Choose the evidence type and use the native visual tools:
   - text: the evidence depends on characters, digits, labels, signs, or documents.
     Use OCR. If the text is tiny or its location is uncertain, first use DINO to
     locate the text carrier and then pass the returned bbox to OCR.
   - visual: the evidence depends on an object, color, shape, count, spatial relation,
     mark, material, or texture. Use DINO only, not OCR.
4. Locate and crop multiple targets independently. Never create one union crop spanning
   separate objects.
5. After every ground_image call, inspect the returned visual preview. If it does not
   contain the requested target, rewrite the query and call the tool again. Confirm the
   evidence visually before returning JSON.

Language and tool contract:
- Write every Analyzer message, tool query, and JSON string in English.
- `ground_image.query` and `grounding_queries` must be short, concrete English noun
  phrases that Grounding DINO can localize. Do not use Chinese or other non-English
  translations for tool queries.
- `visible_focus_instruction` is shown to the Teacher and must be written in English
  while remaining answer-neutral. It may name observable attributes and comparative
  descriptors (including candidate colors or shapes) needed to inspect the image,
  but must not reveal an option letter, assert an answer conclusion, state a reward,
  expose rollout outcomes, or reproduce OCR text.

Return exactly this JSON schema:
{
  "group_summary": "Private summary of the key visual disagreement",
  "crucial_evidence": "Smallest sufficient visual evidence",
  "crucial_evidence_type": "text or visual",
  "tool_route": "ocr or dino",
  "visible_focus_instruction": "Short answer-neutral inspection instruction",
  "grounding_queries": ["one to three concrete English visual targets"],
  "confidence": 0.0
}
Return JSON only, with no Markdown or extra commentary."""


TOOL_USE_APPENDIX = """

The ground_image and read_text tools are Analyzer-only and are never exposed to the
Student. Tool definitions and results are injected by the Qwen native tool-calling
template."""


ONE_SHOT_MESSAGES = [
    {
        "role": "user",
        "content": (
            "Question: What street name is written on the small sign under the bridge?\n\n"
            "Successful rollouts:\n- Zoomed into and read the characters on the sign.\n"
            "- Answered from the visible text.\n\n"
            "Failed rollouts:\n- Guessed from the surrounding road scene.\n"
            "- Answered without reading the sign clearly."
        ),
    },
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "example_ground",
                "type": "function",
                "function": {
                    "name": "ground_image",
                    "arguments": json.dumps(
                        {"query": "small street sign under the bridge", "context_margin": 0.12},
                        ensure_ascii=False,
                    ),
                },
            }
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "example_ground",
        "content": json.dumps(
            {"found": True, "score": 0.84, "bbox": [420, 610, 690, 790]},
            ensure_ascii=False,
        ),
    },
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "example_ocr",
                "type": "function",
                "function": {
                    "name": "read_text",
                    "arguments": json.dumps(
                        {"bbox": [420, 610, 690, 790], "scale": 4},
                        ensure_ascii=False,
                    ),
                },
            }
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "example_ocr",
        "content": json.dumps(
            {"crop_bbox": [420, 610, 690, 790], "text": [{"text": "<OCR result>", "confidence": 0.97}]},
            ensure_ascii=False,
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "group_summary": "Successful rollouts read the sign characters; failed rollouts guessed from scene priors.",
                "crucial_evidence": "Text on the small street sign under the bridge",
                "crucial_evidence_type": "text",
                "tool_route": "ocr",
                "visible_focus_instruction": "Inspect the enlarged small street sign under the bridge and read its characters carefully.",
                "grounding_queries": ["small street sign under the bridge"],
                "confidence": 0.94,
            },
            ensure_ascii=False,
        ),
    },
]


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
    def record(item) -> dict:
        return {
            "rollout_id": item.rollout_id,
            "predicted_label": item.predicted_label,
            "reasoning": item.completion,
        }

    correct = [record(item) for item in group.rollouts if item.reward > 0.5]
    incorrect = [record(item) for item in group.rollouts if item.reward <= 0.5]
    return (
        f"Question:\n{group.question}\n\n"
        f"Successful rollouts:\n{json.dumps(correct, ensure_ascii=False, indent=2)}\n\n"
        f"Failed rollouts:\n{json.dumps(incorrect, ensure_ascii=False, indent=2)}"
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
    bbox_key = "bbox" if tool_name == "ground_image" else "crop_bbox"
    bbox = result.get(bbox_key)
    preview_url = _crop_data_url(image_path, bbox, max_side=max_side)
    if preview_url is None:
        return None
    query = str(arguments.get("query") or result.get("query") or tool_name).strip()
    score = result.get("score")
    score_text = f" with score {float(score):.3f}" if isinstance(score, (int, float)) else ""
    coordinates = ", ".join(str(value) for value in bbox)
    return {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    f"Visual feedback for the `{tool_name}` query `{query}`{score_text}. "
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
        tool_messages = list(messages)
        self.last_tool_trace = []
        for _tool_round in range(self.config.max_tool_rounds):
            body = self._base_body(tool_messages)
            body.update({"tools": registry.schemas, "tool_choice": "auto"})
            message = self._request(body)["choices"][0]["message"]
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                return self._attach_tool_regions(
                    FocusProgram.model_validate(_extract_json(self._message_content(message))),
                    group.image_path,
                    self._ocr_reference_text(group, message),
                )

            tool_messages.append(
                {
                    "role": "assistant",
                    "content": self._message_content(message),
                    "tool_calls": tool_calls,
                }
            )
            for call in tool_calls:
                call_id = str(call.get("id", ""))
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
                self.last_tool_trace.append(
                    {
                        "round": _tool_round + 1,
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
        return self._attach_tool_regions(
            FocusProgram.model_validate(_extract_json(self._message_content(message))),
            group.image_path,
            self._ocr_reference_text(group, message),
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
        """Promote only the final confirmed tool round into private crop regions.

        Earlier tool calls are diagnostic attempts that the Analyzer explicitly
        superseded after seeing their visual feedback.  Exposing them to the
        Teacher would mix failed and successful hypotheses in the privileged
        prefix, so only the latest successful round for the selected route is kept.
        """
        from .schemas import ToolRegion

        ocr_regions: list[tuple[int, ToolRegion]] = []
        dino_regions: list[tuple[int, ToolRegion]] = []
        with Image.open(image_path) as loaded:
            image_size = loaded.size
        for trace in self.last_tool_trace:
            result = trace.get("result", {})
            if not isinstance(result, dict):
                continue
            try:
                trace_round = int(trace.get("round", 0))
            except (TypeError, ValueError):
                trace_round = 0
            if trace.get("name") == "read_text" and isinstance(result.get("crop_bbox"), list):
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
                    ocr_regions.append(
                        (
                            trace_round,
                            ToolRegion(
                                query=query,
                                expanded_box=tuple(int(value) for value in bbox),
                                score=(selected_confidence if tight_bbox is not None else max(confidences, default=0.0)),
                                source="paddle_ocr_text" if tight_bbox is not None else "paddle_ocr_context",
                            ),
                        )
                    )
            elif (
                trace.get("name") == "ground_image"
                and result.get("found", True)
                and isinstance(result.get("bbox"), list)
            ):
                bbox = result["bbox"]
                if len(bbox) == 4:
                    dino_regions.append(
                        (
                            trace_round,
                            ToolRegion(
                                query=str(result.get("query", "Grounding DINO target")),
                                expanded_box=tuple(int(value) for value in bbox),
                                score=float(result.get("score", 0.0)),
                                source="grounding_dino",
                            ),
                        )
                    )

        def final_round(regions: list[tuple[int, ToolRegion]]) -> list[ToolRegion]:
            if not regions:
                return []
            latest = max(round_number for round_number, _region in regions)
            return [region for round_number, region in regions if round_number == latest]

        final_ocr_regions = final_round(ocr_regions)
        final_dino_regions = final_round(dino_regions)
        # The Analyzer's explicit Crucial Evidence decision selects the crop
        # source. Incidental OCR can therefore never replace a visual DINO box.
        if focus.crucial_evidence_type == "text" and focus.tool_route == "ocr":
            selected_regions = final_ocr_regions or final_dino_regions
        else:
            selected_regions = final_dino_regions
        return focus.model_copy(update={"tool_regions": selected_regions})

    def analyze(self, group: GroupRollout) -> FocusProgram:
        content = [
            {"type": "image_url", "image_url": {"url": _data_url(group.image_path)}},
            {"type": "text", "text": build_group_analysis_text(group)},
        ]
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT + (TOOL_USE_APPENDIX if self.config.use_vision_tools else ""),
            },
        ]
        if self.config.use_vision_tools:
            messages.extend(ONE_SHOT_MESSAGES)
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
