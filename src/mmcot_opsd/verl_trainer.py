"""verl trainer extension for online group analysis and visual Teacher evidence."""

from __future__ import annotations

import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from verl.trainer.ppo.ray_trainer import RayPPOTrainer

from .analyzer import OpenAIAnalyzerConfig, OpenAICompatibleAnalyzer
from .evidence import SAFE_FOCUS_FALLBACK, EvidenceBuilderConfig, TeacherEvidenceBuilder, teacher_payload
from .grounding import GroundingDinoConfig, GroundingDinoGrounder
from .reward import extract_option
from .schemas import GroupRollout, Rollout


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class VisualSeedRayPPOTrainer(RayPPOTrainer):
    """Inject a shared hindsight Crop/Zoom prefix into every rollout of an analyzed group.

    This class changes only Teacher input construction.  The actor patch owns the
    objective: ordinary GRPO is evaluated on all rollouts, while SEED OPD is
    evaluated wherever ``self_distillation_mask`` says evidence is available.
    """

    _visual_seed_builder: TeacherEvidenceBuilder | None = None

    def _get_visual_seed_builder(self) -> TeacherEvidenceBuilder:
        if self._visual_seed_builder is None:
            analyzer = OpenAICompatibleAnalyzer(OpenAIAnalyzerConfig.from_env())
            grounder = GroundingDinoGrounder(
                GroundingDinoConfig(
                    model=os.environ.get(
                        "GROUNDING_DINO_MODEL", "IDEA-Research/grounding-dino-base"
                    ),
                    device=os.environ.get("GROUNDING_DINO_DEVICE", "cuda:0"),
                    box_threshold=float(os.environ.get("GROUNDING_DINO_BOX_THRESHOLD", "0.25")),
                    text_threshold=float(os.environ.get("GROUNDING_DINO_TEXT_THRESHOLD", "0.20")),
                    min_short_side=int(os.environ.get("GROUNDING_DINO_MIN_SHORT_SIDE", "768")),
                    local_files_only=_env_bool("GROUNDING_DINO_LOCAL_FILES_ONLY", True),
                )
            )
            evidence_root = Path(
                os.environ.get("VISUAL_SEED_EVIDENCE_DIR", "outputs/visual_seed_evidence")
            ).resolve()
            self._visual_seed_builder = TeacherEvidenceBuilder(
                analyzer,
                grounder,
                EvidenceBuilderConfig(
                    output_dir=evidence_root,
                    mixed_groups_only=_env_bool("VISUAL_SEED_MIXED_GROUPS_ONLY", False),
                    min_rollouts=int(os.environ.get("VISUAL_SEED_MIN_ROLLOUTS", "2")),
                    reuse_cache=_env_bool("VISUAL_SEED_REUSE_CACHE", True),
                ),
            )
        return self._visual_seed_builder

    @staticmethod
    def _extra(batch: Any, index: int) -> dict[str, Any]:
        values = batch.non_tensor_batch.get("extra_info")
        if values is None or values[index] is None:
            return {}
        value = values[index]
        return value if isinstance(value, dict) else dict(value)

    def _decode_rollout(self, batch: Any, index: int) -> str:
        response = batch.batch["responses"][index]
        response_mask = batch.batch["response_mask"][index].bool()
        return self.tokenizer.decode(response[response_mask].detach().cpu(), skip_special_tokens=True)

    def _build_online_teacher_columns(
        self,
        batch: Any,
        reward_tensor: torch.Tensor,
    ) -> dict[str, float]:
        uids = list(batch.non_tensor_batch["uid"])
        grouped_indices: dict[str, list[int]] = defaultdict(list)
        for index, uid in enumerate(uids):
            grouped_indices[str(uid)].append(index)

        batch_size = len(uids)
        teacher_images: list[list[dict[str, Any]]] = [[] for _ in range(batch_size)]
        teacher_prompts: list[list[dict[str, str]]] = [[] for _ in range(batch_size)]
        sequence_rewards = reward_tensor.sum(dim=-1).detach().float().cpu().tolist()
        status_counts = defaultdict(int)
        route_counts = defaultdict(int)
        crop_counts: list[int] = []
        crop_area_fractions: list[float] = []
        crop_scores: list[float] = []
        dino_scores: list[float] = []
        ocr_scores: list[float] = []
        sanitized_count = 0
        correct_counts: list[int] = []
        mixed_group_count = 0
        evidence_build_seconds = 0.0
        builder = self._get_visual_seed_builder()
        teacher_max_image_pixels = int(
            os.environ.get("VISUAL_SEED_TEACHER_MAX_IMAGE_PIXELS", "1048576")
        )
        if teacher_max_image_pixels <= 0:
            raise ValueError("VISUAL_SEED_TEACHER_MAX_IMAGE_PIXELS must be positive")

        for uid, indices in grouped_indices.items():
            first_extra = self._extra(batch, indices[0])
            question = str(first_extra.get("question", "")).strip()
            image_value = first_extra.get("image_path")
            if not question or not image_value:
                raise ValueError(
                    "visual_seed requires extra_info.question and extra_info.image_path in the dataset"
                )
            image_path = Path(str(image_value)).resolve()
            rollouts = []
            for rollout_id, sample_index in enumerate(indices):
                completion = self._decode_rollout(batch, sample_index)
                rollouts.append(
                    Rollout(
                        rollout_id=rollout_id,
                        completion=completion,
                        predicted_label=extract_option(completion),
                        reward=float(sequence_rewards[sample_index]),
                    )
                )
            group = GroupRollout(
                uid=f"step-{self.global_steps:07d}-{uid}",
                question=question,
                image_path=image_path,
                rollouts=rollouts,
            )
            correct_count = sum(item.reward > 0.5 for item in rollouts)
            correct_counts.append(correct_count)
            mixed_group_count += int(0 < correct_count < len(rollouts))
            build_start = time.perf_counter()
            evidence = builder.build(group)
            evidence_build_seconds += time.perf_counter() - build_start
            status_counts[evidence.status] += 1
            if evidence.status == "ready" and evidence.focus is not None:
                route_counts[evidence.focus.crucial_evidence_type] += 1
                crop_counts.append(len(evidence.crops))
                crop_area_fractions.extend(float(crop.area_fraction) for crop in evidence.crops)
                crop_scores.extend(float(crop.score) for crop in evidence.crops)
                sanitized_count += int(evidence.focus.visible_focus_instruction == SAFE_FOCUS_FALLBACK)
                for region in evidence.focus.tool_regions:
                    if region.source == "grounding_dino":
                        dino_scores.append(float(region.score))
                    elif region.source == "paddle_ocr_context":
                        ocr_scores.append(float(region.score))
            prompt, images = teacher_payload(
                evidence,
                question=question,
                max_image_pixels=teacher_max_image_pixels,
            )
            for sample_index in indices:
                teacher_prompts[sample_index] = prompt
                teacher_images[sample_index] = images

        prompt_column = np.empty(batch_size, dtype=object)
        image_column = np.empty(batch_size, dtype=object)
        prompt_column[:] = teacher_prompts
        image_column[:] = teacher_images
        batch.non_tensor_batch["teacher_prompt"] = prompt_column
        batch.non_tensor_batch["visual_seed_teacher_images"] = image_column
        group_count = max(len(grouped_indices), 1)
        ready_count = max(status_counts["ready"], 1)
        metrics = {
            "visual_seed/group_count": float(len(grouped_indices)),
            "visual_seed/evidence_ready_fraction": status_counts["ready"] / group_count,
            "visual_seed/evidence_skipped_fraction": status_counts["skipped"] / group_count,
            "visual_seed/evidence_error_fraction": status_counts["error"] / group_count,
            "visual_seed/route_visual_fraction": route_counts["visual"] / ready_count,
            "visual_seed/route_text_fraction": route_counts["text"] / ready_count,
            "visual_seed/focus_sanitized_fraction": sanitized_count / ready_count,
            "visual_seed/mixed_group_fraction": mixed_group_count / group_count,
            "visual_seed/correct_rollout_fraction": sum(correct_counts) / max(batch_size, 1),
            "timing_s/visual_seed/evidence_build_total": evidence_build_seconds,
            "timing_s/visual_seed/evidence_build_mean": evidence_build_seconds / group_count,
        }
        if crop_counts:
            metrics["visual_seed/crop_count_mean"] = float(np.mean(crop_counts))
            metrics["visual_seed/crop_count_max"] = float(max(crop_counts))
        if crop_area_fractions:
            metrics["visual_seed/crop_area_fraction_mean"] = float(np.mean(crop_area_fractions))
        if crop_scores:
            metrics["visual_seed/crop_score_mean"] = float(np.mean(crop_scores))
        if dino_scores:
            metrics["visual_seed/dino_score_mean"] = float(np.mean(dino_scores))
        if ocr_scores:
            metrics["visual_seed/ocr_score_mean"] = float(np.mean(ocr_scores))
        return metrics

    @staticmethod
    def _reward_component_metrics(reward_extra_infos_dict: dict[str, list] | None) -> dict[str, float]:
        """Expose the two terminal-reward components in the normal trainer logs."""
        if not reward_extra_infos_dict:
            return {}
        metrics = {}
        for key in (
            "answer_reward",
            "format_reward",
            "weighted_answer_reward",
            "weighted_format_reward",
        ):
            values = reward_extra_infos_dict.get(key)
            if values is None:
                continue
            try:
                numeric = np.asarray(values, dtype=np.float32)
            except (TypeError, ValueError):
                continue
            if numeric.size:
                metrics[f"reward/{key}_mean"] = float(numeric.mean())
        return metrics

    def _maybe_build_self_distillation_batch(
        self,
        batch: Any,
        reward_tensor: torch.Tensor,
        reward_extra_infos_dict: dict[str, list] | None = None,
    ):
        loss_mode = self.config.actor_rollout_ref.actor.policy_loss.get("loss_mode", "vanilla")
        if loss_mode != "visual_seed":
            return super()._maybe_build_self_distillation_batch(
                batch, reward_tensor, reward_extra_infos_dict
            )

        online_metrics = self._build_online_teacher_columns(batch, reward_tensor)
        online_metrics.update(self._reward_component_metrics(reward_extra_infos_dict))
        result = super()._maybe_build_self_distillation_batch(
            batch, reward_tensor, reward_extra_infos_dict
        )
        if result is None:
            raise RuntimeError("visual_seed teacher batch construction unexpectedly returned None")
        teacher_batch, metrics = result
        teacher_batch.batch["visual_seed_outcome"] = (
            reward_tensor.sum(dim=-1) > 0.5
        ).to(dtype=torch.float32, device=teacher_batch.batch["self_distillation_mask"].device)
        metrics.update(online_metrics)
        return teacher_batch, metrics
