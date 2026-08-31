"""Build group-level privileged visual evidence for the Teacher branch."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .analyzer import Analyzer
from .grounding import Grounder, crop_tool_regions
from .schemas import GroupRollout, TeacherEvidence


CHOICE_PATTERN = re.compile(r"(?im)^\s*(?:\(([A-D])\)|([A-D])[.)])\s*(.+?)\s*$")


RESPONSE_FORMAT_INSTRUCTION = (
    "Inspect the image carefully and reason only from visible evidence. State one concise, "
    "evidence-first rationale without <think> tags or hidden-chain-of-thought. Do not hedge, "
    "revise, or mention this instruction. On the final line write exactly `FINAL: X`, where X "
    "is the direct final answer. If the question explicitly uses labeled answer choices, X is "
    "the selected choice label."
)

SAFE_FOCUS_FALLBACK = (
    "Inspect the provided zoomed visual evidence and compare only directly visible "
    "details relevant to the question."
)


def _parse_choices(question: str) -> dict[str, str]:
    choices = {}
    for parenthesized, bare, answer_text in CHOICE_PATTERN.findall(question):
        choices[(parenthesized or bare).upper()] = answer_text.strip()
    return choices


def validate_visible_focus(group: GroupRollout, focus_instruction: str) -> None:
    """Reject answer-bearing text while preserving useful visual comparisons.

    Observable attribute words that also occur in multiple-choice options are not
    leakage by themselves.  A focus such as "compare whether the frame is gold or
    bronze" still requires the Teacher to inspect the supplied image and crop.  We
    reject explicit option/answer statements and unqualified assertions instead.
    """
    text = " ".join(focus_instruction.strip().split())
    lowered = text.lower()
    if re.search(
        r"(?i)\b(?:final\s+answer|correct\s+(?:answer|option|choice)|best\s+(?:answer|choice))\s*(?:is|:)"
        r"|\b(?:answer|option|choice)\s*(?:is|:)\s*(?:option\s*)?[A-E](?:\b|\))",
        text,
    ):
        raise ValueError("visible focus instruction contains an answer conclusion")

    successful_labels = {
        str(item.predicted_label).upper()
        for item in group.rollouts
        if item.reward > 0.5 and item.predicted_label
    }
    for label in successful_labels:
        if re.search(rf"(?i)\boption\s*{re.escape(label)}\b", text):
            raise ValueError("visible focus instruction reveals the successful option letter")

    # A bare assertion such as "the object is bronze" leaks a candidate answer,
    # while a comparative inspection instruction such as "compare gold or bronze"
    # is useful visual guidance.  Only inspect answer-choice terms in assertion
    # sentences; ordinary spatial references like "inspect the alpha mark" remain
    # valid.  This intentionally errs toward preserving multimodal prior context.
    choices = _parse_choices(group.question)
    comparison = re.compile(
        r"(?i)\b(?:compare|compared|comparison|whether|versus|rather\s+than|"
        r"relative\s+to|against|distinguish|different|or)\b"
    )
    subjects = (
        r"(?:the|this|that|it|object|item|animal|person|figure|material|color|"
        r"shape|region|poster|frame|hair|coat|surface|mark|image)"
    )
    verbs = r"(?:is|are|was|were|appears?\s+to\s+be|looks?\s+like|shows?|indicates?|identifies?)"
    for answer_text in choices.values():
        normalized_answer = " ".join(answer_text.lower().split()).strip(" .,:;!?`'\"")
        if len(normalized_answer) < 3:
            continue
        term = re.escape(normalized_answer)
        for sentence in re.split(r"[.!?;]", lowered):
            if not re.search(rf"(?<!\w){term}(?!\w)", sentence):
                continue
            if re.search(
                rf"\b(?:answer|option|choice)\s*(?:is|:)\s*(?:option\s*)?(?<!\w){term}(?!\w)",
                sentence,
            ):
                raise ValueError("visible focus instruction contains an answer conclusion")
            if comparison.search(sentence):
                continue
            if re.search(rf"\b{subjects}\b[^,]{{0,60}}\b{verbs}\b[^,]{{0,30}}(?<!\w){term}(?!\w)", sentence):
                raise ValueError("visible focus instruction contains an answer conclusion")


def build_teacher_prompt(question: str, focus_instruction: str, crop_count: int) -> list[dict]:
    content = [
        "<image>",
        question.strip(),
        "\nHindsight visual focus:",
        focus_instruction.strip(),
    ]
    for index in range(crop_count):
        content.extend([f"\nZoomed visual evidence {index + 1}:", "<image>"])
    content.extend(["\nResponse requirements:", RESPONSE_FORMAT_INSTRUCTION])
    return [{"role": "user", "content": "\n".join(content)}]


def fallback_teacher_prompt(question: str) -> list[dict[str, str]]:
    """One-placeholder prompt matching the original image used by verl fallback."""
    return [
        {
            "role": "user",
            "content": f"<image>\n{question.strip()}\n\n{RESPONSE_FORMAT_INSTRUCTION}",
        }
    ]


def teacher_payload(
    evidence: TeacherEvidence,
    *,
    question: str,
    max_image_pixels: int | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Convert cached evidence into Vision-OPD's prompt and image columns."""
    if evidence.status != "ready":
        return fallback_teacher_prompt(question), []

    def image_ref(path: Path) -> dict[str, str | int]:
        result: dict[str, str | int] = {"path": str(path)}
        if max_image_pixels is not None:
            if max_image_pixels <= 0:
                raise ValueError("max_image_pixels must be positive when provided")
            result["max_pixels"] = int(max_image_pixels)
        return result

    images = [image_ref(evidence.original_image_path)]
    images.extend(image_ref(crop.path) for crop in evidence.crops)
    if evidence.teacher_prompt is None:
        prompt = build_teacher_prompt(
            question,
            evidence.focus.visible_focus_instruction if evidence.focus else "Inspect the relevant details.",
            len(evidence.crops),
        )
    else:
        prompt = evidence.teacher_prompt
    return prompt, images


@dataclass(frozen=True)
class EvidenceBuilderConfig:
    output_dir: Path
    mixed_groups_only: bool = False
    min_rollouts: int = 2
    reuse_cache: bool = True


class TeacherEvidenceBuilder:
    def __init__(self, analyzer: Analyzer, grounder: Grounder, config: EvidenceBuilderConfig):
        self.analyzer = analyzer
        self.grounder = grounder
        self.config = config

    def _record_path(self, group: GroupRollout) -> Path:
        safe_uid = "".join(char if char.isalnum() or char in "-_" else "_" for char in group.uid)
        return self.config.output_dir / safe_uid / "evidence.json"

    def build(self, group: GroupRollout) -> TeacherEvidence:
        record_path = self._record_path(group)
        if self.config.reuse_cache and record_path.exists():
            cached = TeacherEvidence.model_validate_json(record_path.read_text(encoding="utf-8"))
            if cached.status != "error":
                return cached

        if len(group.rollouts) < self.config.min_rollouts:
            result = TeacherEvidence(
                uid=group.uid,
                status="skipped",
                original_image_path=group.image_path,
                reason="insufficient_rollouts",
            )
            return self._save(result, record_path)
        if self.config.mixed_groups_only and not group.is_mixed:
            result = TeacherEvidence(
                uid=group.uid,
                status="skipped",
                original_image_path=group.image_path,
                reason="uniform_reward_group",
            )
            return self._save(result, record_path)

        try:
            focus = self.analyzer.analyze(group)
            try:
                validate_visible_focus(group, focus.visible_focus_instruction)
            except ValueError:
                # Analyzer text is privileged and untrusted.  An answer-bearing
                # instruction must never reach the Teacher prompt, but the
                # independently grounded visual evidence can still be useful.
                focus = focus.model_copy(
                    update={"visible_focus_instruction": SAFE_FOCUS_FALLBACK}
                )
                validate_visible_focus(group, focus.visible_focus_instruction)
            if focus.tool_regions:
                crops = crop_tool_regions(group.image_path, focus.tool_regions, record_path.parent)
            else:
                crops = self.grounder.crop_objects(
                    group.image_path,
                    focus,
                    record_path.parent,
                )
            if not crops:
                raise RuntimeError("Grounding DINO produced no object crops")
            result = TeacherEvidence(
                uid=group.uid,
                status="ready",
                focus=focus,
                original_image_path=group.image_path.resolve(),
                crops=crops,
                teacher_prompt=build_teacher_prompt(
                    group.question,
                    focus.visible_focus_instruction,
                    len(crops),
                ),
            )
        except Exception as exc:
            result = TeacherEvidence(
                uid=group.uid,
                status="error",
                original_image_path=group.image_path,
                reason=f"{type(exc).__name__}: {exc}",
            )
        return self._save(result, record_path)

    @staticmethod
    def _save(result: TeacherEvidence, record_path: Path) -> TeacherEvidence:
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_path.write_text(
            json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return result
