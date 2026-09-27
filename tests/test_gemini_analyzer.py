import ast
import importlib.util
import json
import os
from contextlib import redirect_stderr
from io import StringIO
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from PIL import Image

from groove.analyzer import build_group_analysis_text
from groove.evidence import EvidenceBuilderConfig, SAFE_FOCUS_FALLBACK, TeacherEvidenceBuilder
from groove.gemini_analyzer import (
    GEMINI_SYSTEM_PROMPT,
    GeminiAnalyzerConfig,
    GeminiAPIAnalyzer,
    NoDetectorFallback,
    build_gemini_analysis_text,
    gemini_box_to_pixels,
)
from groove.schemas import GroupRollout, Rollout, TeacherEvidence


def crop_call(call_id="call_1", box=None, query="small sign"):
    return {
        "id": call_id, "type": "function",
        "function": {"name": "crop_image", "arguments": json.dumps({
            "query": query, "box_2d": [100, 200, 500, 600] if box is None else box,
        })},
        "extra_content": {"google": {"thought_signature": "test-opaque-signature"}},
    }


def final_message(ids=None, **overrides):
    focus = {
        "group_summary": "Compare the visual target.",
        "visible_focus_instruction": "Inspect the lettering on the sign.",
        "selected_candidate_ids": ["candidate_1"] if ids is None else ids,
    }
    focus.update(overrides)
    return {"role": "assistant", "content": json.dumps(focus)}


class GeminiConfigAndGeometryTest(unittest.TestCase):
    def test_non_square_image_and_full_image_coordinate_conversion(self):
        self.assertEqual(gemini_box_to_pixels([100, 200, 500, 600], (1000, 500)), (200, 50, 600, 250))
        self.assertEqual(gemini_box_to_pixels([0, 0, 1000, 1000], (1000, 500)), (0, 0, 1000, 500))

    def test_rejects_invalid_or_ambiguous_boxes(self):
        for box in (None, [], [0, 1, 2], [0, 1, 2, 3, 4], [0, 0, 0, 5], [500, 0, 100, 5],
                    [0, 0, 1001, 10], [-1, 0, 5, 10], [False, 0, 5, 10], ["0", 0, 5, 10],
                    [0.1, 0.2, 0.5, 0.6], [0, 0, float("nan"), 10]):
            with self.subTest(box=box), self.assertRaises(ValueError):
                gemini_box_to_pixels(box, (1000, 500))

    def test_url_paths_and_secret_repr(self):
        for url, expected in (
            ("https://relay.test/v1", "https://relay.test/v1/"),
            ("https://relay.test/custom/v2/", "https://relay.test/custom/v2/"),
            ("https://relay.test/v1/chat/completions", "https://relay.test/v1/"),
            ("https://generativelanguage.googleapis.com/v1beta/openai/", "https://generativelanguage.googleapis.com/v1beta/openai/"),
        ):
            config = GeminiAnalyzerConfig(base_url=url, api_key="secret-test-key")
            self.assertEqual(config.sdk_base_url, expected)
            self.assertNotIn("secret-test-key", repr(config))
        for url in ("file:///tmp/keys", "https://key@relay.test/v1", "https://relay.test/v1?key=secret"):
            with self.assertRaises(ValueError):
                GeminiAnalyzerConfig(base_url=url, api_key="secret")

    def test_cannot_silently_lower_reasoning_or_remove_budget(self):
        for overrides in ({"reasoning_effort": "low"}, {"max_tool_rounds": 0}, {"max_calls_per_round": 0},
                          {"max_completion_tokens": 0}, {"max_retries": -1}, {"timeout_seconds": float("nan")},
                          {"context_margin": float("nan")}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                GeminiAnalyzerConfig(base_url="https://relay.test/v1", api_key="test", **overrides)

    @unittest.skipUnless(importlib.util.find_spec("dotenv"), "requires gemini-analyzer optional dependencies")
    def test_env_loader_does_not_execute_or_interpolate_or_mutate_environment(self):
        with TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            path = Path(tmp) / ".env"
            path.write_text('OPENAI_BASE_URL="https://relay.test/v1"\nOPENAI_API_KEY=key-${NOT_A_SECRET}\nOPENAI_MODEL=gemini-3.8-flash\n')
            config = GeminiAnalyzerConfig.from_env(path)
            self.assertEqual(config.api_key, "key-${NOT_A_SECRET}")
            self.assertEqual(config.model, "gemini-3.8-flash")
            self.assertEqual(config.reasoning_effort, "high")
            self.assertNotIn("OPENAI_API_KEY", os.environ)
            with patch.dict(os.environ, {"OPENAI_API_KEY": "override"}):
                self.assertEqual(GeminiAnalyzerConfig.from_env(path).api_key, "override")


@unittest.skipUnless(importlib.util.find_spec("openai"), "requires gemini-analyzer optional dependencies")
class GeminiAnalyzerTest(unittest.TestCase):
    def setUp(self):
        import httpx
        from openai import OpenAI

        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.image_path = self.root / "original.png"
        image = Image.new("RGB", (1000, 500), "white")
        image.paste("red", (200, 50, 600, 250))
        image.save(self.image_path)
        self.group = GroupRollout(
            uid="private-rollout-uid", question="What is on the small sign?",
            image_path=self.image_path, ground_truth="PRIVATE_GT_TEXT",
            rollouts=[
                Rollout(rollout_id=173, completion="A multiline\ncorrect trace", predicted_label="PRIVATE_LABEL", is_correct=True),
                Rollout(rollout_id=174, completion="incorrect trace", predicted_label="OTHER_LABEL", is_correct=False),
            ],
        )
        self.responses = []
        self.requests = []

        def respond(request):
            self.assertEqual(str(request.url), "https://relay.test/v1/chat/completions")
            self.assertEqual(request.headers["authorization"], "Bearer test-key")
            self.requests.append(json.loads(request.content))
            response = self.responses.pop(0)
            if "http_error" in response:
                return httpx.Response(response["http_error"], json={"error": {"message": "echo test-key", "type": "bad_request"}})
            message = response.get("message", response)
            return httpx.Response(200, json={
                "id": "chatcmpl-test", "object": "chat.completion", "created": 1, "model": "gemini-3.8-flash",
                "choices": [{"index": 0, "message": message,
                             "finish_reason": response.get("finish_reason", "tool_calls" if message.get("tool_calls") else "stop")}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30,
                          "completion_tokens_details": {"reasoning_tokens": 15}},
            })

        client = OpenAI(api_key="test-key", base_url="https://relay.test/v1", max_retries=0,
                        http_client=httpx.Client(transport=httpx.MockTransport(respond)))
        self.analyzer = GeminiAPIAnalyzer(GeminiAnalyzerConfig(
            base_url="https://relay.test/v1", api_key="test-key", max_tool_rounds=2,
        ), client=client)
        self.addCleanup(self.analyzer.close)

    def tools(self, *calls):
        return {"role": "assistant", "content": None, "tool_calls": list(calls),
                "extra_content": {"google": {"thought_signature": "message-signature"}}}

    def test_end_to_end_preserves_signatures_groups_and_selected_pixels(self):
        self.responses = [self.tools(crop_call(), crop_call("call_2", [0, 0, 1000, 1000], query="full context")), final_message(["candidate_2", "candidate_1"])]
        focus = self.analyzer.analyze(self.group)
        self.assertEqual([r.expanded_box for r in focus.tool_regions], [(0, 0, 1000, 500), (152, 26, 648, 274)])
        self.assertEqual(focus.tool_route, "gemini")
        self.assertEqual(focus.grounding_queries, ["full context", "small sign"])
        self.assertEqual(focus.crucial_evidence_type, "unknown")
        self.assertEqual(focus.crucial_evidence, "")
        self.assertIsNone(focus.confidence)
        self.assertTrue(all(r.source == "gemini_native_bbox" for r in focus.tool_regions))
        for request in self.requests:
            self.assertEqual(request["messages"][0], {"role": "system", "content": GEMINI_SYSTEM_PROMPT})
            self.assertEqual(request["reasoning_effort"], "high")
            self.assertEqual(request["max_tokens"], 16384)
            self.assertNotIn("chat_template_kwargs", request)
            self.assertEqual([t["function"]["name"] for t in request["tools"]], ["crop_image"])
        input_text = self.requests[0]["messages"][1]["content"][1]["text"]
        group_input = json.loads(input_text)
        self.assertEqual(set(group_input), {"question", "ground_truth", "successful_reasoning", "failed_reasoning"})
        self.assertEqual(group_input["ground_truth"], "PRIVATE_GT_TEXT")
        self.assertEqual(group_input["successful_reasoning"], ["A multiline\ncorrect trace"])
        self.assertNotIn("PRIVATE_LABEL", input_text)
        self.assertNotIn(self.group.uid, input_text)
        messages = self.requests[1]["messages"]
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "tool", "tool", "user", "user"])
        self.assertEqual(messages[2]["extra_content"]["google"]["thought_signature"], "message-signature")
        self.assertEqual(messages[2]["tool_calls"][0]["extra_content"]["google"]["thought_signature"], "test-opaque-signature")
        self.assertTrue(messages[5]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))
        audit = self.analyzer.last_api_trace
        self.assertEqual(audit[0]["response"]["usage"]["completion_tokens_details"]["reasoning_tokens"], 15)
        self.assertNotIn("base64,", json.dumps(audit))
        self.assertNotIn("test-key", json.dumps(audit))

    def test_ground_truth_is_required_before_api_calls_and_legacy_input_stays_private(self):
        for value in (None, "", " \n "):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "nonempty ground_truth"):
                self.analyzer.analyze(self.group.model_copy(update={"ground_truth": value}))
        self.assertEqual(self.requests, [])
        legacy_input = json.loads(build_group_analysis_text(self.group))
        self.assertNotIn("ground_truth", legacy_input)
        self.assertNotIn("PRIVATE_GT_TEXT", json.dumps(legacy_input))
        before = self.group.model_dump_json()
        gemini_input = json.loads(build_gemini_analysis_text(self.group))
        self.assertEqual(gemini_input["failed_reasoning"], ["incorrect trace"])
        self.assertEqual(self.group.model_dump_json(), before)

    def test_repairs_extra_metadata_instead_of_trusting_model_authored_routes(self):
        self.responses = [self.tools(crop_call()), final_message(tool_route="ocr", grounding_queries=["invented"]), final_message()]
        focus = self.analyzer.analyze(self.group)
        self.assertEqual(focus.tool_route, "gemini")
        self.assertEqual(focus.grounding_queries, ["small sign"])
        self.assertEqual(len(self.requests), 3)
        repair = self.requests[-1]["messages"][-1]["content"]
        self.assertIn("Return only group_summary", repair)
        self.assertNotIn("tool_route=gemini", repair)

    def test_blank_diagnosis_and_rule_are_not_accepted(self):
        for field in ("group_summary", "visible_focus_instruction"):
            self.responses = [self.tools(crop_call()), final_message(**{field: " \n "}), final_message()]
            with self.subTest(field=field):
                focus = self.analyzer.analyze(self.group)
                self.assertTrue(getattr(focus, field).strip())

    def test_bad_box_is_returned_as_tool_error_and_can_be_corrected(self):
        self.responses = [self.tools(crop_call(box=[0, 0, 1001, 1000])), self.tools(crop_call("call_2")), final_message(["candidate_2"])]
        focus = self.analyzer.analyze(self.group)
        self.assertEqual(focus.selected_candidate_ids, ["candidate_2"])
        self.assertIn("error", self.analyzer.last_tool_trace[0]["result"])
        self.assertFalse(self.analyzer.last_tool_trace[0]["visual_feedback_attached"])
        self.assertEqual(self.requests[-1]["tool_choice"], "none")

    def test_repairs_missing_selection_once_without_more_tools(self):
        self.responses = [self.tools(crop_call()), final_message([]), final_message()]
        self.assertEqual(self.analyzer.analyze(self.group).selected_candidate_ids, ["candidate_1"])
        self.assertEqual(len(self.requests), 3)
        self.assertEqual(self.requests[-1]["tool_choice"], "none")
        self.assertTrue(all(r["reasoning_effort"] == "high" for r in self.requests))

    def test_rejects_hallucinated_and_failed_candidates_after_repair(self):
        self.responses = [self.tools(crop_call(), crop_call("bad", [0, 0, 0, 1])), final_message(["candidate_2"]), final_message(["made_up"])]
        with self.assertRaises(ValueError):
            self.analyzer.analyze(self.group)
        self.assertEqual(len(self.requests), 3)

    def test_bounds_tool_calls_and_never_uses_external_detector(self):
        self.responses = [self.tools(crop_call("a"), crop_call("b"), crop_call("c"), crop_call("d")), final_message()]
        self.analyzer.analyze(self.group)
        self.assertEqual(len(self.analyzer.last_tool_trace), 4)
        self.assertIn("error", self.analyzer.last_tool_trace[3]["result"])
        self.assertEqual([m["role"] for m in self.requests[1]["messages"]][3:7], ["tool"] * 4)

    def test_unknown_tool_is_rejected(self):
        call = crop_call()
        call["function"]["name"] = "ground_image"
        self.responses = [self.tools(call), final_message()]
        with self.assertRaisesRegex(ValueError, "no selectable"):
            self.analyzer.analyze(self.group)
        self.assertIn("Only crop_image", self.analyzer.last_tool_trace[0]["result"]["error"])

    def test_truncated_completion_is_not_accepted_even_if_json_is_complete(self):
        self.responses = [self.tools(crop_call()), {"message": final_message(), "finish_reason": "length"}]
        with self.assertRaisesRegex(ValueError, "truncated"):
            self.analyzer.analyze(self.group)
        self.assertEqual(len(self.requests), 2)

    def test_api_errors_do_not_expose_credentials(self):
        self.responses = [{"http_error": 400}]
        with self.assertRaises(RuntimeError) as raised:
            self.analyzer.analyze(self.group)
        self.assertNotIn("test-key", str(raised.exception))
        self.assertNotIn("test-key", json.dumps(self.analyzer.last_api_trace))

    def test_same_outcome_groups_are_still_analyzed_and_state_is_reset(self):
        self.responses = [self.tools(crop_call()), final_message()]
        self.analyzer.analyze(self.group)
        uniform = self.group.model_copy(update={"rollouts": [r.model_copy(update={"is_correct": False}) for r in self.group.rollouts]})
        self.responses = [self.tools(crop_call()), final_message()]
        focus = self.analyzer.analyze(uniform)
        self.assertEqual(focus.selected_candidate_ids, ["candidate_1"])
        self.assertEqual(len(self.analyzer.last_tool_trace), 1)
        group_input = json.loads(self.requests[2]["messages"][1]["content"][1]["text"])
        self.assertEqual(group_input["successful_reasoning"], [])
        self.assertEqual(len(group_input["failed_reasoning"]), 2)

    def test_materializes_compatible_evidence_and_sanitizes_answer_leakage(self):
        self.responses = [self.tools(crop_call()), final_message(
            group_summary="The answer reference is PRIVATE_GT_TEXT, but the label is noisy.",
            visible_focus_instruction="The correct answer is A.",
        )]
        output_dir = self.root / "evidence"
        builder = TeacherEvidenceBuilder(self.analyzer, NoDetectorFallback(), EvidenceBuilderConfig(output_dir=output_dir))
        evidence = builder.build(self.group)
        self.assertEqual(evidence.status, "ready")
        self.assertEqual(evidence.focus.visible_focus_instruction, SAFE_FOCUS_FALLBACK)
        self.assertEqual(len(evidence.tool_trace), 1)
        self.assertTrue(evidence.crops[0].path.is_file())
        record = next(output_dir.glob("*/evidence.json"))
        self.assertEqual(TeacherEvidence.model_validate_json(record.read_text()).focus.tool_route, "gemini")
        teacher_text = json.dumps(evidence.teacher_prompt)
        self.assertNotIn("correct trace", teacher_text)
        self.assertNotIn("crop_image", teacher_text)
        self.assertNotIn("box_2d", teacher_text)
        self.assertNotIn("PRIVATE_GT_TEXT", teacher_text)
        self.assertNotIn("label is noisy", teacher_text)
        restored = TeacherEvidence.model_validate_json(record.read_text())
        self.assertEqual(restored.focus.crucial_evidence_type, "unknown")
        self.assertIsNone(restored.focus.confidence)
        self.assertIn("PRIVATE_GT_TEXT", restored.focus.group_summary)

    def test_shared_rule_reaches_teacher_without_diagnosis_or_ground_truth(self):
        rule = "Locate the target, separate its lettering from its background, then verify the requested attribute."
        self.responses = [self.tools(crop_call()), final_message(
            group_summary="PRIVATE_GT_TEXT exposes a false positive in the labels.",
            visible_focus_instruction=rule,
        )]
        builder = TeacherEvidenceBuilder(self.analyzer, NoDetectorFallback(),
                                         EvidenceBuilderConfig(output_dir=self.root / "rule"))
        evidence = builder.build(self.group)
        self.assertEqual(evidence.status, "ready")
        teacher_text = json.dumps(evidence.teacher_prompt)
        self.assertIn(rule, teacher_text)
        self.assertNotIn("PRIVATE_GT_TEXT", teacher_text)
        self.assertNotIn("false positive", teacher_text)

    def test_no_evidence_error_preserves_trace_without_detector_fallback(self):
        self.responses = [self.tools(crop_call(box=[0, 0, 0, 1])), final_message()]
        builder = TeacherEvidenceBuilder(self.analyzer, NoDetectorFallback(), EvidenceBuilderConfig(output_dir=self.root / "failure"))
        evidence = builder.build(self.group)
        self.assertEqual(evidence.status, "error")
        self.assertEqual(len(evidence.tool_trace), 1)
        self.assertIsNone(evidence.teacher_prompt)

    def test_prompt_is_one_standalone_literal_with_complete_evidence_contract(self):
        path = Path(__file__).resolve().parents[1] / "src/groove/gemini_analyzer.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        definitions = [
            node.value for node in tree.body if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "GEMINI_SYSTEM_PROMPT" for target in node.targets)
        ]
        self.assertEqual(len(definitions), 1)
        self.assertIsInstance(definitions[0], ast.Constant)
        self.assertEqual(definitions[0].value, GEMINI_SYSTEM_PROMPT)
        self.assertNotIn("SYSTEM_PROMPT", [
            alias.name for node in tree.body if isinstance(node, ast.ImportFrom) for alias in node.names
        ])
        self.assertLessEqual(len(GEMINI_SYSTEM_PROMPT.split()), 200)
        self.assertIn("ground-truth answer as the primary reference", GEMINI_SYSTEM_PROMPT)
        self.assertIn("success/failure labels may be wrong", GEMINI_SYSTEM_PROMPT)
        self.assertIn("one shared verification rule", GEMINI_SYSTEM_PROMPT)
        normalized = " ".join(GEMINI_SYSTEM_PROMPT.split())
        self.assertIn("Select the smallest sufficient evidence covering all targets needed to resolve the main reasoning issue.", normalized)
        self.assertIn("A correct final answer does not guarantee correct reasoning; never invent visual evidence to fit the ground truth.", normalized)
        self.assertIn("Inspect every returned preview", GEMINI_SYSTEM_PROMPT)
        self.assertNotIn("reassess the outcome labels", GEMINI_SYSTEM_PROMPT)
        for field in ("crucial_evidence", "crucial_evidence_type", "tool_route", "grounding_queries", "confidence"):
            self.assertNotIn(f'"{field}"', GEMINI_SYSTEM_PROMPT)
        self.assertNotIn("ground_image", GEMINI_SYSTEM_PROMPT)
        self.assertNotIn("read_text", GEMINI_SYSTEM_PROMPT)
        self.assertNotRegex(GEMINI_SYSTEM_PROMPT, r"[\u3400-\u9fff]")

    def _cli(self):
        path = Path(__file__).resolve().parents[1] / "scripts/analyze_gemini_rollouts.py"
        spec = importlib.util.spec_from_file_location("gemini_offline_cli", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_cli_records_evidence_and_audit_without_keys_and_refuses_overwrite(self):
        cli = self._cli()
        groups_path = self.root / "groups.jsonl"
        group = self.group.model_copy(update={"image_path": Path("original.png")})
        groups_path.write_text(group.model_dump_json() + "\n")
        output = self.root / "cli-result"
        self.responses = [self.tools(crop_call()), final_message()]
        with patch.object(cli.GeminiAnalyzerConfig, "from_env", return_value=self.analyzer.config), \
                patch.object(cli, "GeminiAPIAnalyzer", return_value=self.analyzer):
            self.assertEqual(cli.main(["--groups", str(groups_path), "--output-dir", str(output)]), 0)
        files = list(output.rglob("*.json"))
        self.assertTrue(any(p.name == "evidence.json" for p in files))
        self.assertTrue(any(p.name == "api_trace.json" for p in files))
        self.assertTrue(all("test-key" not in p.read_text() for p in files))
        self.assertEqual(json.loads((output / "summary.json").read_text())["ready"], 1)
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(manifest["analysis_protocol"], "ground-truth-shared-rule-v1")
        self.assertEqual(len(manifest["system_prompt_sha256"]), 64)
        saved_group = json.loads((output / "group-000001/group.json").read_text())
        self.assertEqual(saved_group["ground_truth"], "PRIVATE_GT_TEXT")
        before = (output / "manifest.json").read_bytes()
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as raised:
            cli.main(["--groups", str(groups_path), "--output-dir", str(output)])
        self.assertEqual(raised.exception.code, 2)
        self.assertEqual((output / "manifest.json").read_bytes(), before)

    def test_cli_dry_run_does_not_construct_analyzer_or_write(self):
        cli = self._cli()
        groups_path = self.root / "groups.jsonl"
        groups_path.write_text(self.group.model_dump_json() + "\n")
        output = self.root / "dry-run"
        with patch.object(cli.GeminiAnalyzerConfig, "from_env", return_value=self.analyzer.config), \
                patch.object(cli, "GeminiAPIAnalyzer") as constructor:
            self.assertEqual(cli.main(["--groups", str(groups_path), "--output-dir", str(output), "--dry-run"]), 0)
            constructor.assert_not_called()
        self.assertFalse(output.exists())

    def test_cli_focus_uses_selected_gemini_boxes_and_records_rendering_settings(self):
        cli = self._cli()
        groups_path = self.root / "focus-groups.jsonl"
        groups_path.write_text(self.group.model_dump_json() + "\n")
        output = self.root / "cli-focus"
        self.responses = [self.tools(crop_call()), final_message()]
        with patch.object(cli.GeminiAnalyzerConfig, "from_env", return_value=self.analyzer.config), \
                patch.object(cli, "GeminiAPIAnalyzer", return_value=self.analyzer):
            result = cli.main(["--groups", str(groups_path), "--output-dir", str(output),
                               "--teacher-evidence-mode", "focus", "--focus-blur-alpha", "0.5"])
        self.assertEqual(result, 0)
        evidence = TeacherEvidence.model_validate_json(next(output.rglob("evidence.json")).read_text())
        self.assertEqual(evidence.image_config.mode, "focus")
        self.assertEqual(evidence.focus_image.boxes, [(152, 26, 648, 274)])
        self.assertEqual(evidence.image_paths, [evidence.focus_image.path])
        self.assertEqual(len(evidence.tool_trace), 1)
        self.assertNotIn("box_2d", str(evidence.teacher_prompt))
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(manifest["image_config"], {"mode": "focus", "blur_alpha": 0.5, "blur_radius": 12.0})

    def test_cli_refuses_unaligned_training_dump(self):
        cli = self._cli()
        path = self.root / "raw.jsonl"
        path.write_text(json.dumps({"input": "question", "output": "answer", "accuracy": 1, "gts": "private"}))
        with self.assertRaisesRegex(ValueError, "GroupRollout") as raised:
            cli.load_groups(path)
        self.assertNotIn("private", str(raised.exception))

    def test_cli_rejects_missing_ground_truth_before_creating_output(self):
        cli = self._cli()
        path = self.root / "missing-gt.jsonl"
        group = self.group.model_dump(mode="json")
        del group["ground_truth"]
        path.write_text(json.dumps(group))
        output = self.root / "missing-gt-result"
        with patch.object(cli.GeminiAnalyzerConfig, "from_env", return_value=self.analyzer.config), \
                patch.object(cli, "GeminiAPIAnalyzer") as constructor, \
                self.assertRaisesRegex(ValueError, "Line 1 requires a nonempty ground_truth"):
            cli.main(["--groups", str(path), "--output-dir", str(output), "--dry-run"])
        constructor.assert_not_called()
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
