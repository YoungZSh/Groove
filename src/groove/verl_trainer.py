"""verl trainer extension for online group analysis and visual Teacher evidence."""

from __future__ import annotations

import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.model import compute_position_id_with_mask

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

    @staticmethod
    def _normalize_teacher_image(image: Any) -> Image.Image:
        max_pixels = image.get("max_pixels") if isinstance(image, dict) else None
        if isinstance(image, Image.Image):
            normalized = image.convert("RGB")
        elif isinstance(image, (str, os.PathLike)):
            with Image.open(image) as value:
                normalized = value.convert("RGB")
        elif isinstance(image, dict):
            if image.get("image") is not None:
                normalized = GrooveRayPPOTrainer._normalize_teacher_image(image["image"])
            elif image.get("bytes") is not None:
                normalized = Image.open(BytesIO(image["bytes"])).convert("RGB")
            elif image.get("path"):
                normalized = GrooveRayPPOTrainer._normalize_teacher_image(image["path"])
            else:
                raise TypeError(f"Unsupported teacher image dictionary: {image.keys()}")
        else:
            raise TypeError(f"Unsupported teacher image type: {type(image)}")

        if max_pixels is None or normalized.width * normalized.height <= int(max_pixels):
            return normalized
        scale = (int(max_pixels) / (normalized.width * normalized.height)) ** 0.5
        size = (max(1, round(normalized.width * scale)), max(1, round(normalized.height * scale)))
        return normalized.resize(size, Image.Resampling.LANCZOS)

    @staticmethod
    def _teacher_images_available(images: Any) -> bool:
        if images is None:
            return False
        values = images.tolist() if isinstance(images, np.ndarray) else images
        if not isinstance(values, (list, tuple)):
            values = [values]
        return any(value is not None for value in values)

    @staticmethod
    def _extract_images_from_messages(messages: list[dict]) -> list[Image.Image]:
        images = []
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict) or item.get("type") != "image":
                    continue
                images.append(GrooveRayPPOTrainer._normalize_teacher_image(item))
        return images

    @staticmethod
    def _resize_teacher_images(images: list[Image.Image], scale: float) -> list[Image.Image]:
        if scale >= 0.999:
            return images
        return [
            image.resize(
                (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                Image.Resampling.LANCZOS,
            )
            for image in images
        ]

    @staticmethod
    def _replace_teacher_message_images(messages: list[dict], images: list[Image.Image]) -> list[dict]:
        updated = deepcopy(messages)
        image_offset = 0
        for message in updated:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict) or item.get("type") != "image":
                    continue
                if image_offset >= len(images):
                    raise ValueError("Teacher image count is smaller than the message placeholders")
                item.pop("path", None)
                item.pop("bytes", None)
                item.pop("max_pixels", None)
                item["image"] = images[image_offset]
                image_offset += 1
        if image_offset != len(images):
            raise ValueError("Teacher image count does not match the message placeholders")
        return updated

    def _build_teacher_messages_from_template(
        self,
        messages: list[dict],
        images: list[Any],
    ) -> list[dict]:
        normalized = [self._normalize_teacher_image(image) for image in images]
        updated = deepcopy(messages)
        image_offset = 0
        for message in updated:
            if not isinstance(message.get("content"), str):
                continue
            parts = []
            for segment in filter(None, re.split(r"(<image>)", message["content"])):
                if segment == "<image>":
                    if image_offset >= len(normalized):
                        raise ValueError("Teacher image count is smaller than the prompt placeholders")
                    parts.append({"type": "image", "image": normalized[image_offset]})
                    image_offset += 1
                else:
                    parts.append({"type": "text", "text": segment})
            message["content"] = parts
        if image_offset != len(normalized):
            raise ValueError("Teacher image count does not match the prompt placeholders")
        return updated

    def _process_teacher_multimodal_prompt(
        self,
        messages: list[dict],
        prompt_images: list[Image.Image],
        apply_kwargs: dict[str, Any],
        max_prompt_len: int,
    ) -> tuple[str, dict[str, torch.Tensor]]:
        """Resize, never tokenizer-truncate, multimodal Teacher prefixes."""
        candidate_images = list(prompt_images)
        candidate_messages = self._replace_teacher_message_images(messages, candidate_images)
        last_prompt_len = 0
        for _ in range(8):
            raw_prompt = self.processor.apply_chat_template(
                candidate_messages,
                tokenize=False,
                add_generation_prompt=True,
                **apply_kwargs,
            )
            model_inputs = dict(
                self.processor(
                    text=[raw_prompt],
                    images=candidate_images or None,
                    videos=None,
                    return_tensors="pt",
                    truncation=False,
                )
            )
            last_prompt_len = int(model_inputs["input_ids"].shape[-1])
            if last_prompt_len <= max_prompt_len:
                return raw_prompt, model_inputs
            scale = min((max_prompt_len / last_prompt_len) ** 0.5 * 0.97, 0.90)
            next_images = self._resize_teacher_images(candidate_images, scale)
            if all(a.size == b.size for a, b in zip(next_images, candidate_images, strict=True)):
                break
            candidate_images = next_images
            candidate_messages = self._replace_teacher_message_images(messages, candidate_images)
        raise ValueError(f"Teacher prompt exceeds max_prompt_len: {last_prompt_len} > {max_prompt_len}")

    def _build_teacher_prefix_inputs(
        self,
        messages: list[dict],
        max_prompt_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        apply_kwargs = dict(self.config.data.apply_chat_template_kwargs or {})
        prompt_images = self._extract_images_from_messages(messages)
        _raw_prompt, model_inputs = self._process_teacher_multimodal_prompt(
            messages,
            prompt_images,
            apply_kwargs,
            max_prompt_len,
        )
        multi_modal_inputs = model_inputs.copy()
        prompt_input_ids = multi_modal_inputs.pop("input_ids").squeeze(0)
        prompt_attention_mask = multi_modal_inputs.pop("attention_mask").squeeze(0)

        if hasattr(self.processor, "get_rope_index"):
            model_type = getattr(getattr(self.processor, "config", None), "model_type", None)
            if model_type in {"qwen3_5", "qwen3_5_moe", "qwen3_vl", "qwen3_vl_moe"}:
                token_types = multi_modal_inputs.pop("mm_token_type_ids", None)
                if token_types is None:
                    token_types = torch.zeros_like(prompt_input_ids).unsqueeze(0)
                    token_types[0][prompt_input_ids == self.processor.image_token_id] = 1
                position_ids = self.processor.get_rope_index(
                    input_ids=prompt_input_ids.unsqueeze(0),
                    mm_token_type_ids=token_types,
                    image_grid_thw=multi_modal_inputs.get("image_grid_thw"),
                    video_grid_thw=multi_modal_inputs.get("video_grid_thw"),
                    attention_mask=prompt_attention_mask.unsqueeze(0),
                )
            else:
                position_ids = self.processor.get_rope_index(
                    input_ids=prompt_input_ids.unsqueeze(0),
                    image_grid_thw=multi_modal_inputs.get("image_grid_thw"),
                    video_grid_thw=multi_modal_inputs.get("video_grid_thw"),
                    attention_mask=prompt_attention_mask.unsqueeze(0),
                )
            if isinstance(position_ids, tuple):
                position_ids = position_ids[0]
            if position_ids.dim() == 3 and position_ids.shape[1] == 1:
                position_ids = position_ids.squeeze(1)
            if model_type in {"qwen3_5", "qwen3_5_moe"} and position_ids.shape[0] == 3:
                text_positions = torch.arange(prompt_input_ids.shape[-1]).unsqueeze(0)
                position_ids = torch.cat((text_positions.to(position_ids), position_ids), dim=0)
        else:
            position_ids = compute_position_id_with_mask(prompt_attention_mask.unsqueeze(0)).squeeze(0)
        return prompt_input_ids, prompt_attention_mask, position_ids, multi_modal_inputs

    def _build_groove_teacher_batch(
        self,
        batch: DataProto,
    ) -> tuple[DataProto, torch.Tensor, dict[str, float]]:
        config = self.config.groove
        teacher_key = config.get("teacher_image_key", "groove_teacher_images")
        batch_size = len(batch)
        cache: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]] = {}
        prefixes = []
        evidence_mask = []
        for index in range(batch_size):
            images = batch.non_tensor_batch[teacher_key][index]
            images = images.tolist() if isinstance(images, np.ndarray) else list(images or [])
            has_evidence = self._teacher_images_available(images)
            evidence_mask.append(float(has_evidence))
            uid = str(batch.non_tensor_batch.get("uid", np.arange(batch_size))[index])
            if uid not in cache:
                if has_evidence:
                    template = list(batch.non_tensor_batch["teacher_prompt"][index])
                    messages = self._build_teacher_messages_from_template(template, images)
                else:
                    messages = list(batch.non_tensor_batch["raw_prompt"][index])
                cache[uid] = self._build_teacher_prefix_inputs(
                    messages,
                    int(config.get("max_reprompt_len", self.config.data.max_prompt_length)),
                )
            prefixes.append(cache[uid])

        max_prefix = max(int(item[0].shape[-1]) for item in prefixes)
        response = batch.batch["responses"]
        response_mask = batch.batch["response_mask"]
        pad_token_id = self.tokenizer.pad_token_id or 0
        prompt_ids = torch.full((batch_size, max_prefix), pad_token_id, dtype=response.dtype)
        prompt_mask = torch.zeros((batch_size, max_prefix), dtype=batch.batch["attention_mask"].dtype)
        rope_dims = prefixes[0][2].shape[0] if prefixes[0][2].dim() == 2 else None
        if rope_dims is None:
            prompt_positions = torch.zeros((batch_size, max_prefix), dtype=prefixes[0][2].dtype)
        else:
            prompt_positions = torch.zeros((batch_size, rope_dims, max_prefix), dtype=prefixes[0][2].dtype)
        response_positions = []
        multi_modal_inputs = np.empty(batch_size, dtype=object)
        for index, (ids, mask, positions, mm_inputs) in enumerate(prefixes):
            length = ids.shape[-1]
            prompt_ids[index, -length:] = ids
            prompt_mask[index, -length:] = mask
            if rope_dims is None:
                prompt_positions[index, -length:] = positions
                start = positions[-1]
                response_positions.append(torch.arange(response.shape[-1], dtype=positions.dtype) + start + 1)
            else:
                prompt_positions[index, :, -length:] = positions
                response_positions.append(
                    torch.arange(response.shape[-1], dtype=positions.dtype).unsqueeze(0) + positions[:, -1:] + 1
                )
            multi_modal_inputs[index] = mm_inputs
        response_position_ids = torch.stack(response_positions)
        full_position_ids = torch.cat((prompt_positions, response_position_ids), dim=-1)
        teacher_batch = DataProto.from_dict(
            tensors={
                "prompts": prompt_ids,
                "responses": response.cpu(),
                "input_ids": torch.cat((prompt_ids, response.cpu()), dim=-1),
                "attention_mask": torch.cat((prompt_mask, response_mask.cpu()), dim=-1),
                "position_ids": full_position_ids,
                "response_mask": response_mask.cpu(),
            },
            non_tensors={"multi_modal_inputs": multi_modal_inputs},
        )
        teacher_batch.meta_info = dict(batch.meta_info)
        metrics = {
            "groove/teacher_prefix_cache_entries": float(len(cache)),
            "groove/teacher_evidence_fraction": float(np.mean(evidence_mask)),
        }
        return teacher_batch, torch.tensor(evidence_mask, dtype=torch.float32), metrics

    def _postprocess_advantages(
        self,
        batch: DataProto,
        reward_tensor: torch.Tensor,
        reward_extra_infos_dict: dict[str, list] | None = None,
    ) -> tuple[DataProto, dict[str, float]]:
        metrics = self._reward_component_metrics(reward_extra_infos_dict)
        groove_config = self.config.get("groove", {}) or {}
        if not bool(groove_config.get("enabled", False)):
            return batch, metrics

        metrics.update(self._build_online_teacher_columns(batch, reward_tensor))
        teacher_batch, evidence_mask, teacher_metrics = self._build_groove_teacher_batch(batch)
        metrics.update(teacher_metrics)
        started = time.perf_counter()
        teacher_output, _teacher_mfu = self._compute_old_log_prob(teacher_batch)
        metrics["timing_s/groove/teacher_log_prob"] = time.perf_counter() - started

        from groove.losses import combine_grpo_opsd_advantages, groove_opsd_advantages

        student_log_probs = batch.batch["old_log_probs"]
        teacher_log_probs = teacher_output.batch["old_log_probs"].to(student_log_probs.device)
        opsd_advantages, opsd_metrics = groove_opsd_advantages(
            student_log_probs,
            teacher_log_probs,
            batch.batch["response_mask"],
            evidence_mask=evidence_mask.to(student_log_probs.device),
            advantage_clip=groove_config.get("opsd_advantage_clip"),
        )
        batch.batch["advantages"] = combine_grpo_opsd_advantages(
            batch.batch["advantages"],
            opsd_advantages,
            opsd_coef=float(groove_config.get("opsd_advantage_coef", 0.01)),
        )
        metrics.update({f"actor/groove_opsd_{key}": value for key, value in opsd_metrics.__dict__.items()})
        return batch, metrics
