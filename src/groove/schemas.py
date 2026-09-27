"""Typed records passed between rollout analysis, grounding, and verl."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Rollout(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rollout_id: int
    completion: str
    predicted_label: str | None
    is_correct: bool


class GroupRollout(BaseModel):
    model_config = ConfigDict(extra="forbid")

    uid: str
    question: str
    image_path: Path
    rollouts: list[Rollout]
    # Optional for legacy analyzers; required by the GT-assisted Gemini path.
    ground_truth: str | None = None

    @property
    def is_mixed(self) -> bool:
        outcomes = {item.is_correct for item in self.rollouts}
        return len(outcomes) > 1


class InstanceBox(BaseModel):
    """One detector candidate in original-image pixel coordinates, not a GT label."""

    model_config = ConfigDict(extra="forbid")

    bbox: tuple[float, float, float, float]
    score: float = Field(ge=0.0, le=1.0)

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, values):
        if not all(math.isfinite(value) for value in values):
            raise ValueError("instance bbox coordinates must be finite")
        x1, y1, x2, y2 = values
        if x1 >= x2 or y1 >= y2:
            raise ValueError("instance bbox must have positive area")
        return values


class ToolRegion(BaseModel):
    """Private Analyzer-selected region, never inserted as Teacher text."""

    query: str
    expanded_box: tuple[int, int, int, int]
    score: float = Field(ge=0.0, le=1.0)
    source: str
    kind: Literal["crop", "instance_boxes"] = "crop"
    instances: list[InstanceBox] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_instances(self):
        if (self.kind == "instance_boxes") != bool(self.instances):
            raise ValueError("instance_boxes evidence requires nonempty instances; crops must not have instances")
        return self


class FocusProgram(BaseModel):
    """Analyzer output. Object phrases stay tool-private."""

    model_config = ConfigDict(extra="forbid")

    group_summary: str
    crucial_evidence: str = ""
    crucial_evidence_type: Literal["text", "visual", "unknown"] = "visual"
    tool_route: Literal["ocr", "dino", "gemini"] = "dino"
    visible_focus_instruction: str
    grounding_queries: list[str] = Field(min_length=1, max_length=3)
    selected_candidate_ids: list[str] = Field(default_factory=list, max_length=3)
    context_margin: float = Field(default=0.12, ge=0.10, le=0.15)
    confidence: float | None = Field(default=0.5, ge=0.0, le=1.0)
    tool_regions: list[ToolRegion] = Field(default_factory=list)

    @field_validator("grounding_queries")
    @classmethod
    def normalize_queries(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        for value in values:
            query = " ".join(value.strip().split())
            if query and query not in normalized:
                normalized.append(query)
        if not normalized:
            raise ValueError("At least one non-empty grounding query is required")
        return normalized

    @field_validator("selected_candidate_ids")
    @classmethod
    def normalize_candidate_ids(cls, values: list[str]) -> list[str]:
        normalized = []
        for value in values:
            candidate_id = str(value).strip()
            if candidate_id and candidate_id not in normalized:
                normalized.append(candidate_id)
        return normalized


class ObjectCrop(BaseModel):
    """A selected Teacher image; the legacy name also covers full-frame overlays."""
    model_config = ConfigDict(extra="forbid")

    query: str
    score: float
    raw_box: tuple[float, float, float, float]
    expanded_box: tuple[int, int, int, int]
    path: Path
    area_fraction: float
    kind: Literal["crop", "instance_boxes"] = "crop"
    instances: list[InstanceBox] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_instances(self):
        if (self.kind == "instance_boxes") != bool(self.instances):
            raise ValueError("instance_boxes image requires nonempty instances; crops must not have instances")
        return self


class TeacherEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    uid: str
    status: Literal["ready", "skipped", "error"]
    focus: FocusProgram | None = None
    original_image_path: Path
    crops: list[ObjectCrop] = Field(default_factory=list)
    teacher_prompt: list[dict] | None = None
    reason: str | None = None
    tool_trace: list[dict] = Field(default_factory=list)

    @model_validator(mode="after")
    def ready_requires_evidence(self) -> "TeacherEvidence":
        if self.status == "ready" and (not self.crops or not self.teacher_prompt):
            raise ValueError("ready evidence requires crops and teacher_prompt")
        return self
