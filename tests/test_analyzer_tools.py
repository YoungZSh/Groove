import json
import unittest
from pathlib import Path

from groove.analyzer import (
    OpenAIAnalyzerConfig,
    OpenAICompatibleAnalyzer,
    SYSTEM_PROMPT,
    _crop_data_url,
    _tight_ocr_text_bbox,
    _tool_feedback_message,
    build_group_analysis_text,
)
from groove.analyzer_tools import ANALYZER_TOOL_SCHEMAS, AnalyzerVisionToolRegistry
from groove.schemas import FocusProgram, GroupRollout, Rollout


class AnalyzerToolsTest(unittest.TestCase):
    def test_schemas_expose_grounding_and_ocr_only(self):
        self.assertEqual(
            [item["function"]["name"] for item in ANALYZER_TOOL_SCHEMAS],
            ["ground_image", "read_text"],
        )

    def test_bbox_validation_clamps_to_image(self):
        self.assertEqual(
            AnalyzerVisionToolRegistry._bbox([-1, 2, 99, 200], (50, 100)),
            (0.0, 2.0, 50.0, 100.0),
        )
        with self.assertRaises(ValueError):
            AnalyzerVisionToolRegistry._bbox([8, 4, 8, 9], (50, 100))

    def test_tight_ocr_bbox_ignores_punctuation_noise(self):
        result = {
            "crop_bbox": [0, 0, 1044, 1311],
            "text": [
                {"text": "-", "bbox": [[198, 394], [200, 394], [200, 396], [198, 396]]},
                {"text": "35.000", "bbox": [[70.5, 544.25], [99.25, 542.75], [99.5, 551.75], [71, 553.25]]},
            ],
        }

        self.assertEqual(_tight_ocr_text_bbox(result), (70.5, 542.75, 99.5, 553.25))

    def test_tight_ocr_bbox_prefers_text_matching_private_reference(self):
        result = {
            "crop_bbox": [0, 0, 2000, 1500],
            "text": [
                {
                    "text": "TechnipFMC",
                    "confidence": 0.95,
                    "bbox": [[1392, 906], [1428, 906], [1428, 914], [1392, 914]],
                },
                {
                    "text": "DHPENEWS",
                    "confidence": 0.54,
                    "bbox": [[678, 931], [701, 931], [701, 936], [678, 936]],
                },
            ],
        }

        self.assertEqual(
            _tight_ocr_text_bbox(result, reference_text="Deep Energy vessel name"),
            (678.0, 931.0, 701.0, 936.0),
        )

    def test_analyzer_uses_tight_ocr_text_box_for_teacher_crop(self):
        from tempfile import TemporaryDirectory
        from PIL import Image

        with TemporaryDirectory() as directory:
            image_path = Path(directory) / "image.jpg"
            Image.new("RGB", (1000, 1000), color=(10, 20, 30)).save(image_path)
            analyzer = OpenAICompatibleAnalyzer(
                OpenAIAnalyzerConfig(base_url="http://localhost", api_key="test")
            )
            analyzer.last_tool_trace = [
                {
                    "round": 1,
                    "candidate_id": "candidate_1",
                    "name": "read_text",
                    "result": {
                        "crop_bbox": [0, 0, 900, 900],
                        "text": [
                            {
                                "text": "35.000",
                                "confidence": 0.95,
                                "bbox": [[100, 200], [200, 200], [200, 240], [100, 240]],
                            }
                        ],
                    },
                }
            ]
            focus = FocusProgram(
                group_summary="text",
                visible_focus_instruction="Inspect the text.",
                grounding_queries=["orange sign"],
                crucial_evidence_type="text",
                tool_route="ocr",
                selected_candidate_ids=["candidate_1"],
                context_margin=0.12,
            )

            updated = analyzer._attach_tool_regions(focus, image_path)

            self.assertEqual(len(updated.tool_regions), 1)
            self.assertEqual(updated.tool_regions[0].expanded_box, (88, 195, 212, 245))
            self.assertEqual(updated.tool_regions[0].source, "paddle_ocr_text")

    def test_group_text_is_programmatically_split_without_reward_values(self):
        group = GroupRollout(
            uid="split",
            question="question",
            image_path=Path("image.jpg"),
            rollouts=[
                Rollout(rollout_id=0, completion="correct trace", predicted_label="A", is_correct=True),
                Rollout(rollout_id=1, completion="wrong trace", predicted_label="B", is_correct=False),
            ],
        )

        text = build_group_analysis_text(group)
        payload = json.loads(text)

        self.assertEqual(
            set(payload),
            {"question", "successful_reasoning", "failed_reasoning"},
        )
        self.assertEqual(payload["question"], "question")
        self.assertEqual(payload["successful_reasoning"], ["correct trace"])
        self.assertEqual(payload["failed_reasoning"], ["wrong trace"])
        self.assertNotIn('"reward"', text)
        self.assertNotIn("rollout_id", text)
        self.assertNotIn("predicted_label", text)

    def test_analyzer_prompt_is_english(self):
        self.assertNotRegex(SYSTEM_PROMPT, r"[\u3400-\u9fff]")
        self.assertIn("short English noun phrase", SYSTEM_PROMPT)

        for schema in ANALYZER_TOOL_SCHEMAS:
            description = schema["function"].get("description", "")
            self.assertNotRegex(description, r"[\u3400-\u9fff]")

    def test_tool_loop_returns_grounding_crop_to_analyzer(self):
        from tempfile import TemporaryDirectory
        from unittest.mock import patch
        from PIL import Image

        class FakeRegistry:
            schemas = []

            def __init__(self):
                self.calls = []

            def execute(self, image_path, name, arguments):
                self.calls.append((image_path, name, arguments))
                bbox = [0, 0, 4, 4] if arguments["query"] == "wrong sign" else [4, 3, 30, 18]
                return {
                    "found": True,
                    "query": arguments["query"],
                    "score": 0.8,
                    "bbox": bbox,
                    "image_size": [40, 20],
                }

        class FakeAnalyzer(OpenAICompatibleAnalyzer):
            def __init__(self, config):
                super().__init__(config)
                self.requests = []

            def _request(self, body):
                # The analyzer mutates its message list between rounds; snapshot
                # the list so assertions can inspect each request as sent.
                self.requests.append({**body, "messages": list(body.get("messages", []))})
                if len(self.requests) == 1:
                    return {
                        "choices": [
                            {
                                "message": {
                                    "content": "",
                                    "tool_calls": [
                                        {
                                            "id": "call-1",
                                            "function": {
                                                "name": "ground_image",
                                                "arguments": '{"query":"wrong sign"}',
                                            },
                                        }
                                    ],
                                }
                            }
                        ]
                    }
                if len(self.requests) == 2:
                    return {
                        "choices": [
                            {
                                "message": {
                                    "content": "",
                                    "tool_calls": [
                                        {
                                            "id": "call-2",
                                            "function": {
                                                "name": "ground_image",
                                                "arguments": '{"query":"small blue sign"}',
                                            },
                                        }
                                    ],
                                }
                            }
                        ]
                    }
                return {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    '{"group_summary":"A sign differs.",'
                                    '"crucial_evidence":"The sign",'
                                    '"crucial_evidence_type":"visual",'
                                    '"tool_route":"dino",'
                                    '"visible_focus_instruction":"Inspect the sign.",'
                                    '"grounding_queries":["small blue sign"],'
                                    '"selected_candidate_ids":["candidate_2"],'
                                    '"confidence":0.8}'
                                )
                            }
                        }
                    ]
                }

        with TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "image.jpg"
            Image.new("RGB", (40, 20), color=(10, 20, 30)).save(image_path)
            group = GroupRollout(
                uid="feedback",
                question="What is visible?",
                image_path=image_path,
                rollouts=[
                    Rollout(rollout_id=0, completion="trace", predicted_label="A", is_correct=True),
                ],
            )
            analyzer = FakeAnalyzer(
                OpenAIAnalyzerConfig(
                    base_url="http://localhost",
                    api_key="test",
                    max_tool_rounds=3,
                    tool_feedback_max_side=32,
                )
            )
            with patch("groove.analyzer_tools.AnalyzerVisionToolRegistry", FakeRegistry):
                focus = analyzer._analyze_with_tools(group, [])

            self.assertEqual(focus.grounding_queries, ["small blue sign"])
            self.assertEqual(len(analyzer.requests), 3)
            second_messages = analyzer.requests[1]["messages"]
            feedback = [
                message
                for message in second_messages
                if message.get("role") == "user" and isinstance(message.get("content"), list)
            ]
            self.assertEqual(len(feedback), 1)
            self.assertEqual(feedback[0]["content"][1]["type"], "image_url")
            self.assertTrue(analyzer.last_tool_trace[0]["visual_feedback_attached"])
            self.assertEqual([trace["round"] for trace in analyzer.last_tool_trace], [1, 2])
            self.assertEqual(
                [trace["candidate_id"] for trace in analyzer.last_tool_trace],
                ["candidate_1", "candidate_2"],
            )
            self.assertEqual(len(focus.tool_regions), 1)
            self.assertEqual(focus.tool_regions[0].query, "small blue sign")
            self.assertEqual(focus.tool_regions[0].expanded_box, (4, 3, 30, 18))

    def test_analyzer_honors_multiple_selected_candidates_without_iou_deduplication(self):
        from tempfile import TemporaryDirectory
        from PIL import Image

        with TemporaryDirectory() as directory:
            image_path = Path(directory) / "image.jpg"
            Image.new("RGB", (40, 30), color=(10, 20, 30)).save(image_path)
            analyzer = OpenAICompatibleAnalyzer(
                OpenAIAnalyzerConfig(base_url="http://localhost", api_key="test")
            )
            analyzer.last_tool_trace = [
                {
                    "candidate_id": "candidate_1",
                    "name": "ground_image",
                    "arguments": {"query": "first target"},
                    "result": {"found": True, "query": "first target", "bbox": [2, 2, 25, 25], "score": 0.8},
                },
                {
                    "candidate_id": "candidate_2",
                    "name": "ground_image",
                    "arguments": {"query": "second target"},
                    "result": {"found": True, "query": "second target", "bbox": [3, 3, 24, 24], "score": 0.7},
                },
            ]
            focus = FocusProgram(
                group_summary="Two targets are needed.",
                visible_focus_instruction="Inspect both targets.",
                grounding_queries=["first target", "second target"],
                selected_candidate_ids=["candidate_2", "candidate_1"],
            )

            updated = analyzer._attach_tool_regions(focus, image_path)

            self.assertEqual(
                [region.query for region in updated.tool_regions],
                ["second target", "first target"],
            )

    def test_analyzer_rejects_an_unavailable_candidate(self):
        from tempfile import TemporaryDirectory
        from PIL import Image

        with TemporaryDirectory() as directory:
            image_path = Path(directory) / "image.jpg"
            Image.new("RGB", (40, 30), color=(10, 20, 30)).save(image_path)
            analyzer = OpenAICompatibleAnalyzer(
                OpenAIAnalyzerConfig(base_url="http://localhost", api_key="test")
            )
            analyzer.last_tool_trace = []
            focus = FocusProgram(
                group_summary="Missing target.",
                visible_focus_instruction="Inspect the target.",
                grounding_queries=["target"],
                selected_candidate_ids=["candidate_99"],
            )

            with self.assertRaisesRegex(ValueError, "unavailable candidate"):
                analyzer._attach_tool_regions(focus, image_path)

    def test_analyzer_repairs_a_missing_final_selection_once(self):
        from tempfile import TemporaryDirectory
        from PIL import Image

        with TemporaryDirectory() as directory:
            image_path = Path(directory) / "image.jpg"
            Image.new("RGB", (40, 30), color=(10, 20, 30)).save(image_path)
            group = GroupRollout(
                uid="repair",
                question="What is shown?",
                image_path=image_path,
                rollouts=[Rollout(rollout_id=0, completion="trace", predicted_label=None, is_correct=True)],
            )
            analyzer = OpenAICompatibleAnalyzer(
                OpenAIAnalyzerConfig(base_url="http://localhost", api_key="test")
            )
            analyzer.last_tool_trace = [
                {
                    "candidate_id": "candidate_1",
                    "name": "ground_image",
                    "arguments": {"query": "target"},
                    "result": {"found": True, "query": "target", "bbox": [2, 2, 25, 25], "score": 0.8},
                }
            ]
            repaired = (
                '{"group_summary":"Target found.",'
                '"visible_focus_instruction":"Inspect the target.",'
                '"grounding_queries":["target"],'
                '"selected_candidate_ids":["candidate_1"]}'
            )
            requests = []

            def repair_request(body):
                requests.append(body)
                return {"choices": [{"message": {"content": repaired}}]}

            analyzer._request = repair_request
            initial = {
                "content": (
                    '{"group_summary":"Target found.",'
                    '"visible_focus_instruction":"Inspect the target.",'
                    '"grounding_queries":["target"]}'
                )
            }

            focus = analyzer._finalize_focus_selection(group, initial, [])

            self.assertEqual(focus.selected_candidate_ids, ["candidate_1"])
            self.assertEqual(focus.tool_regions[0].query, "target")
            self.assertEqual(len(requests), 1)
            self.assertIn("candidate_1", requests[0]["messages"][-1]["content"])

    def test_grounding_rejects_cjk_queries(self):
        registry = object.__new__(AnalyzerVisionToolRegistry)
        with self.assertRaisesRegex(ValueError, "English query"):
            registry.ground_image(Path("missing.jpg"), "桥下的路牌")

    def test_tool_feedback_contains_a_multimodal_crop(self):
        with self.subTest("crop data URL"):
            from tempfile import TemporaryDirectory
            from PIL import Image
            import base64
            from io import BytesIO

            with TemporaryDirectory() as directory:
                image_path = Path(directory) / "image.jpg"
                Image.new("RGB", (40, 20), color=(20, 40, 60)).save(image_path)
                data_url = _crop_data_url(image_path, [5, 4, 30, 18], max_side=32)
                self.assertIsNotNone(data_url)
                encoded = data_url.split(",", 1)[1]
                self.assertEqual(base64.b64decode(encoded)[:2], b"\xff\xd8")

                message = _tool_feedback_message(
                    image_path,
                    "ground_image",
                    {"query": "small blue sign"},
                    {
                        "candidate_id": "candidate_1",
                        "query": "small blue sign",
                        "bbox": [5, 4, 30, 18],
                        "score": 0.8,
                    },
                    max_side=32,
                )
                self.assertEqual(message["role"], "user")
                self.assertIsInstance(message["content"], list)
                self.assertEqual(message["content"][0]["type"], "text")
                self.assertEqual(message["content"][1]["type"], "image_url")
                self.assertIn("exact local crop", message["content"][0]["text"])
                self.assertIn("candidate_1", message["content"][0]["text"])


if __name__ == "__main__":
    unittest.main()
