#!/usr/bin/env python3
"""Unit tests for pure GC-Visual-SEED probe helpers."""

from __future__ import annotations

import unittest

from group_visual_opvd_probe import (
    analysis_leakage_flags,
    build_messages,
    expand_box,
    extract_label,
    group_stats,
    parse_analysis_response,
    selected_group,
    select_grounding_boxes,
)


class GroupVisualProbeTests(unittest.TestCase):
    def test_extract_label_prefers_final_marker(self) -> None:
        self.assertEqual(extract_label("A may fit, but evidence favors B.\nFINAL: C"), "C")
        self.assertEqual(extract_label("Therefore the answer is (B)."), "B")

    def test_mixed_group_selection(self) -> None:
        rollouts = [
            {"predicted_label": "A", "reward": 1},
            {"predicted_label": "B", "reward": 0},
        ] * 4
        group = {"rollouts": rollouts, "stats": group_stats(rollouts)}
        keep, reason = selected_group(group, "mixed", 8)
        self.assertTrue(keep)
        self.assertEqual(reason, "mixed_outcome")

    def test_parse_fenced_analysis(self) -> None:
        raw = """```json
        {
          "group_summary": "successes inspect the target",
          "success_common_evidence": "a small emblem",
          "failure_common_pattern": "background distractors",
          "visible_focus_instruction": "Inspect and compare the sail emblems.",
          "grounding_queries": ["blue sail"],
          "spatial_selector": {"type": "ordinal_x", "arguments": {"rank": 2}},
          "crop_policy": {"context_margin": 0.25, "max_crops": 1},
          "supporting_success_ids": [0],
          "contrasting_failure_ids": [1],
          "confidence": 0.8
        }
        ```"""
        parsed = parse_analysis_response(raw)
        self.assertEqual(parsed["grounding_queries"], ["blue sail"])
        self.assertEqual(parsed["spatial_selector"]["type"], "ordinal_x")

    def test_ordinal_grounding_selection(self) -> None:
        candidates = [
            {"query_index": 0, "score": 0.9, "box": [300, 0, 400, 100]},
            {"query_index": 0, "score": 0.8, "box": [100, 0, 200, 100]},
            {"query_index": 0, "score": 0.7, "box": [500, 0, 600, 100]},
        ]
        selected = select_grounding_boxes(
            candidates,
            {"type": "ordinal_x", "arguments": {"rank": 2, "direction": "left_to_right"}},
        )
        self.assertEqual(selected[0]["box"], [300, 0, 400, 100])

    def test_per_query_keeps_separate_objects(self) -> None:
        candidates = [
            {"query_index": 0, "score": 0.9, "box": [0, 0, 10, 10]},
            {"query_index": 0, "score": 0.5, "box": [20, 0, 30, 10]},
            {"query_index": 1, "score": 0.8, "box": [100, 0, 110, 10]},
        ]
        selected = select_grounding_boxes(
            candidates, {"type": "per_query", "arguments": {}}
        )
        self.assertEqual([item["box"] for item in selected], [
            [0, 0, 10, 10],
            [100, 0, 110, 10],
        ])

    def test_message_contains_one_image_slot_per_crop(self) -> None:
        messages = build_messages("question", evidence_count=3)
        image_slots = [
            item for item in messages[0]["content"] if item["type"] == "image"
        ]
        self.assertEqual(len(image_slots), 4)  # original plus three crops

    def test_expand_box_is_clamped(self) -> None:
        self.assertEqual(expand_box([0, 0, 100, 100], (200, 150), 0.25), [0, 0, 125, 125])

    def test_leakage_a_is_not_triggered_by_article(self) -> None:
        analysis = {
            "visible_focus_instruction": "Inspect a small emblem on the sail."
        }
        question = "Which emblem?\n(A) star\n(B) moon"
        self.assertEqual(analysis_leakage_flags(analysis, question, "A"), [])
        analysis["visible_focus_instruction"] = "The correct option is A."
        flags = analysis_leakage_flags(analysis, question, "A")
        self.assertIn("answer_language", flags)
        self.assertIn("gold_option_letter", flags)


if __name__ == "__main__":
    unittest.main()
