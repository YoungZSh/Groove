"""verl trainer extension for online group analysis and visual Teacher evidence."""

from __future__ import annotations

import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
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


class GrooveRayPPOTrainer(RayPPOTrainer):
    """Inject a shared hindsight Crop/Zoom prefix into every rollout of an analyzed group.

    This class changes only Teacher input construction.  The actor patch owns the
    objective: ordinary GRPO credit is built for all rollouts, while signed
    visual OPSD credit is added wherever ``self_distillation_mask`` says
    evidence is available.  The actor sends the combined advantage through one
    PPO objective.
    """

    _groove_builder: TeacherEvidenceBuilder | None = None

    @staticmethod
    def _remote_tool_endpoints_configured() -> bool:
        """Return whether both auxiliary visual tools are on remote HTTP workers.

        A parallel builder must not instantiate multiple local GroundingDINO
        models.  The online launcher uses both URLs, so this guard keeps the
        safe serial fallback for local/offline probe configurations.
        """
        return bool(
            os.environ.get("ANALYZER_GROUNDING_URL", "").strip()
            and os.environ.get("ANALYZER_OCR_URL", "").strip()
        )

    def _new_groove_builder(self) -> TeacherEvidenceBuilder:
        """Create an independent builder for one concurrent group request.

        ``OpenAICompatibleAnalyzer`` keeps a per-request tool trace, so each
        concurrent group gets its own Analyzer instance.  Evidence output paths
        are UID-scoped and therefore safe to write concurrently.
        """
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
            os.environ.get("GROOVE_EVIDENCE_DIR", "outputs/groove")
        ).resolve()
        return TeacherEvidenceBuilder(
            analyzer,
            grounder,
            EvidenceBuilderConfig(
                output_dir=evidence_root,
                mixed_groups_only=_env_bool("GROOVE_MIXED_GROUPS_ONLY", False),
                min_rollouts=int(os.environ.get("GROOVE_MIN_ROLLOUTS", "2")),
                reuse_cache=_env_bool("GROOVE_REUSE_CACHE", True),
            ),
        )

    def _get_groove_builder(self) -> TeacherEvidenceBuilder:
        if self._groove_builder is None:
            self._groove_builder = self._new_groove_builder()
        return self._groove_builder

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
        evidence_build_wall_seconds = 0.0
        teacher_max_image_pixels = int(
            os.environ.get("GROOVE_TEACHER_MAX_IMAGE_PIXELS", "1048576")
        )
        if teacher_max_image_pixels <= 0:
            raise ValueError("GROOVE_TEACHER_MAX_IMAGE_PIXELS must be positive")

        group_jobs: list[tuple[str, list[int], str, GroupRollout]] = []
        for uid, indices in grouped_indices.items():
            first_extra = self._extra(batch, indices[0])
            question = str(first_extra.get("question", "")).strip()
            image_value = first_extra.get("image_path")
            if not question or not image_value:
                raise ValueError(
                    "groove requires extra_info.question and extra_info.image_path in the dataset"
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

            group_jobs.append((str(uid), indices, question, group))

        requested_workers = int(os.environ.get("GROOVE_MAX_CONCURRENCY", "8"))
        if requested_workers <= 0:
            raise ValueError("GROOVE_MAX_CONCURRENCY must be positive")
        # The remote Analyzer vLLM instance advertises max-num-seqs=16, so
        # eight in-flight groups are within its scheduler budget.  DINO/OCR are
        # deliberately single-worker GPU services; they queue tool requests,
        # while the Analyzer requests themselves can overlap safely because each
        # job owns its mutable tool trace.  Keep local/offline runs serial to
        # avoid loading one local DINO model per thread.
        parallel_tools = self._remote_tool_endpoints_configured() and _env_bool(
            "ANALYZER_USE_VISION_TOOLS", False
        )
        worker_count = min(requested_workers, len(group_jobs)) if parallel_tools else 1

        def build_one(job: tuple[str, list[int], str, GroupRollout]):
            _uid, _indices, _question, group = job
            builder = (
                self._new_groove_builder()
                if worker_count > 1
                else self._get_groove_builder()
            )
            build_start = time.perf_counter()
            result = builder.build(group)
            return result, time.perf_counter() - build_start

        wall_start = time.perf_counter()
        if worker_count > 1:
            with ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="visual-evidence",
            ) as executor:
                built_results = list(executor.map(build_one, group_jobs))
        else:
            built_results = [build_one(job) for job in group_jobs]
        evidence_build_wall_seconds = time.perf_counter() - wall_start

        for (uid, indices, question, _group), (evidence, build_seconds) in zip(
            group_jobs, built_results, strict=True
        ):
            evidence_build_seconds += build_seconds
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
                    elif region.source in {"paddle_ocr_context", "paddle_ocr_text"}:
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
        batch.non_tensor_batch["groove_teacher_images"] = image_column
        group_count = max(len(grouped_indices), 1)
        ready_count = max(status_counts["ready"], 1)
        metrics = {
            "groove/group_count": float(len(grouped_indices)),
            "groove/evidence_ready_fraction": status_counts["ready"] / group_count,
            "groove/evidence_skipped_fraction": status_counts["skipped"] / group_count,
            "groove/evidence_error_fraction": status_counts["error"] / group_count,
            "groove/route_visual_fraction": route_counts["visual"] / ready_count,
            "groove/route_text_fraction": route_counts["text"] / ready_count,
            "groove/focus_sanitized_fraction": sanitized_count / ready_count,
            "groove/mixed_group_fraction": mixed_group_count / group_count,
            "groove/correct_rollout_fraction": sum(correct_counts) / max(batch_size, 1),
            "timing_s/groove/evidence_build_total": evidence_build_seconds,
            "timing_s/groove/evidence_build_mean": evidence_build_seconds / group_count,
            "timing_s/groove/evidence_build_wall": evidence_build_wall_seconds,
            "groove/evidence_concurrency": float(worker_count),
        }
        if crop_counts:
            metrics["groove/crop_count_mean"] = float(np.mean(crop_counts))
            metrics["groove/crop_count_max"] = float(max(crop_counts))
        if crop_area_fractions:
            metrics["groove/crop_area_fraction_mean"] = float(np.mean(crop_area_fractions))
        if crop_scores:
            metrics["groove/crop_score_mean"] = float(np.mean(crop_scores))
        if dino_scores:
            metrics["groove/dino_score_mean"] = float(np.mean(dino_scores))
        if ocr_scores:
            metrics["groove/ocr_score_mean"] = float(np.mean(ocr_scores))
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
        # Vanilla GRPO has no self-distillation inputs at all. Return before
        # delegating to verl so the base helper cannot inspect or construct any
        # Teacher/evidence columns on the ablation path.
        if loss_mode == "vanilla":
            return None
        if loss_mode != "groove":
            return super()._maybe_build_self_distillation_batch(
                batch, reward_tensor, reward_extra_infos_dict
            )

        online_metrics = self._build_online_teacher_columns(batch, reward_tensor)
        online_metrics.update(self._reward_component_metrics(reward_extra_infos_dict))
        result = super()._maybe_build_self_distillation_batch(
            batch, reward_tensor, reward_extra_infos_dict
        )
        if result is None:
            raise RuntimeError("groove teacher batch construction unexpectedly returned None")
        teacher_batch, metrics = result
        teacher_batch.batch["groove_outcome"] = (
            reward_tensor.sum(dim=-1) > 0.5
        ).to(dtype=torch.float32, device=teacher_batch.batch["self_distillation_mask"].device)
        metrics.update(online_metrics)
        return teacher_batch, metrics
