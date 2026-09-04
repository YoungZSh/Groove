"""Typed records passed between rollout analysis, grounding, and verl."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Rollout(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rollout_id: int
    completion: str
    predicted_label: str | None
    reward: float


class GroupRollout(BaseModel):
    model_config = ConfigDict(extra="forbid")

    uid: str
    question: str
    image_path: Path
    rollouts: list[Rollout]

    @property
    def is_mixed(self) -> bool:
        outcomes = {float(item.reward) > 0.5 for item in self.rollouts}
        return len(outcomes) > 1


class ToolRegion(BaseModel):
    """Private Analyzer-selected region, never inserted as Teacher text."""

    query: str
    expanded_box: tuple[int, int, int, int]
    score: float = Field(ge=0.0, le=1.0)
    source: str


class FocusProgram(BaseModel):
    """Analyzer output. Object phrases stay tool-private."""

    model_config = ConfigDict(extra="forbid")

    group_summary: str
    crucial_evidence: str = ""
    crucial_evidence_type: Literal["text", "visual"] = "visual"
    tool_route: Literal["ocr", "dino"] = "dino"
    visible_focus_instruction: str
    grounding_queries: list[str] = Field(min_length=1, max_length=3)
    context_margin: float = Field(default=0.12, ge=0.10, le=0.15)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
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


class ObjectCrop(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    score: float
    raw_box: tuple[float, float, float, float]
    expanded_box: tuple[int, int, int, int]
    path: Path
    area_fraction: float


class TeacherEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    uid: str
    status: Literal["ready", "skipped", "error"]
    focus: FocusProgram | None = None
    original_image_path: Path
    crops: list[ObjectCrop] = Field(default_factory=list)
    teacher_prompt: list[dict] | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def ready_requires_evidence(self) -> "TeacherEvidence":
        if self.status == "ready" and (not self.crops or not self.teacher_prompt):
            raise ValueError("ready evidence requires crops and teacher_prompt")
        return self
