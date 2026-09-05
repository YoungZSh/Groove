from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from groove.deepeyes_reward import (
    _judge_one,
    compute_score,
    extract_answer,
    find_inner_repetition,
    parse_judgement,
)


class DeepEyesRewardLoopTest(unittest.TestCase):
    def test_scalar_reward_adapter_forwards_question_and_answers(self):
        expected = {"score": 1.0, "accuracy": 1.0}
        with patch("groove.deepeyes_reward._judge_one", return_value=expected) as judge:
            result = compute_score(
                data_source="deepeyes_vstar_grpo",
                solution_str="<answer>green</answer>",
                ground_truth="The kite is green.",
                extra_info={"question": "What color is the kite?"},
            )

        self.assertEqual(result, expected)
        judge.assert_called_once_with(
            "What color is the kite?",
            "The kite is green.",
            "<answer>green</answer>",
            answer_reward_weight=1.0,
            format_reward_weight=0.2,
        )

    def test_scalar_reward_adapter_requires_question(self):
        with self.assertRaises(ValueError):
            compute_score("source", "answer", "ground truth", {})

    def test_remote_judge_is_constrained_to_a_binary_choice(self):
        response = MagicMock()
        response.__enter__.return_value = response
        payload = {"choices": [{"message": {"content": "1"}}]}
        environment = {
            "DEEPEYES_JUDGE_API_KEY": "test-key",
            "DEEPEYES_JUDGE_MAX_RETRIES": "0",
        }
        with (
            patch.dict("os.environ", environment, clear=False),
            patch("groove.deepeyes_reward.urllib.request.urlopen", return_value=response) as urlopen,
            patch("groove.deepeyes_reward.json.load", return_value=payload),
        ):
            result = _judge_one("What color?", "green", "<answer>green</answer>")

        body = json.loads(urlopen.call_args.args[0].data)
        self.assertEqual(body["messages"][0]["content"], "You are a helpful assistant.")
        self.assertEqual(body["structured_outputs"], {"choice": ["0", "1"]})
        self.assertEqual(body["max_completion_tokens"], 4)
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["format_reward"], 0.0)

    def test_bare_correct_answer_gets_deepeyes_format_penalty(self):
        response = MagicMock()
        response.__enter__.return_value = response
        payload = {"choices": [{"message": {"content": "1"}}]}
        environment = {
            "DEEPEYES_JUDGE_API_KEY": "test-key",
            "DEEPEYES_JUDGE_MAX_RETRIES": "0",
            "DEEPEYES_REPETITION_ZERO_REWARD": "false",
        }
        with (
            patch.dict("os.environ", environment, clear=False),
            patch("groove.deepeyes_reward.urllib.request.urlopen", return_value=response),
            patch("groove.deepeyes_reward.json.load", return_value=payload),
        ):
            result = _judge_one("What color?", "green", "green")

        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["format_valid"], 0.0)
        self.assertEqual(result["format_reward"], -1.0)
        self.assertAlmostEqual(result["score"], 0.8)

    def test_wrong_malformed_answer_gets_negative_format_penalty(self):
        response = MagicMock()
        response.__enter__.return_value = response
        payload = {"choices": [{"message": {"content": "0"}}]}
        environment = {
            "DEEPEYES_JUDGE_API_KEY": "test-key",
            "DEEPEYES_JUDGE_MAX_RETRIES": "0",
            "DEEPEYES_REPETITION_ZERO_REWARD": "false",
        }
        with (
            patch.dict("os.environ", environment, clear=False),
            patch("groove.deepeyes_reward.urllib.request.urlopen", return_value=response),
            patch("groove.deepeyes_reward.json.load", return_value=payload),
        ):
            result = _judge_one("What color?", "green", "blue")

        self.assertEqual(result["accuracy"], 0.0)
        self.assertEqual(result["format_reward"], -1.0)
        self.assertAlmostEqual(result["score"], -0.2)

    def test_format_requires_one_nonempty_terminal_answer_pair(self):
        self.assertEqual(extract_answer(" <answer>green</answer> "), ("green", True))
        self.assertEqual(extract_answer("green"), ("green", False))
        self.assertEqual(extract_answer("<answer> </answer>"), ("", False))
        self.assertEqual(
            extract_answer("<answer>blue</answer><answer>green</answer>"),
            ("green", False),
        )
        self.assertEqual(
            extract_answer("reasoning <answer>green</answer>"),
            ("green", True),
        )

    def test_reasoning_then_answer_keeps_full_reward_and_judges_only_final_answer(self):
        response = MagicMock()
        response.__enter__.return_value = response
        payload = {"choices": [{"message": {"content": "1"}}]}
        reasoning = "I first considered blue, but the visible surface is green."
        with (
            patch.dict("os.environ", {"DEEPEYES_JUDGE_API_KEY": "test-key",
                                      "DEEPEYES_REPETITION_ZERO_REWARD": "false"}),
            patch("groove.deepeyes_reward.urllib.request.urlopen", return_value=response) as urlopen,
            patch("groove.deepeyes_reward.json.load", return_value=payload),
        ):
            result = _judge_one("What color?", "green", reasoning + "\n<answer>green</answer>")
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["format_valid"], 1.0)
        judge_input = json.loads(urlopen.call_args.args[0].data)["messages"][1]["content"]
        self.assertNotIn(reasoning, judge_input)
        self.assertIn("[Model_answer]: green\nJudgement:", judge_input)

    def test_terminal_answer_rejects_nested_unbalanced_and_noncanonical_tags(self):
        invalid = [
            "Reason. <answer>green</answer> extra text",
            "Reason. <answer>green<answer>green</answer>",
            "Reason. </answer><answer>green</answer>",
            "Reason. <answer>green</answer></answer>",
            "<ANSWER>blue</ANSWER><answer>green</answer>",
            "Reason. <ANSWER>green</ANSWER>",
            "Reason. <answer class='final'>green</answer>",
            "Reason. <answer>green",
            "Reason. <answer> \n </answer>",
            "Reason. <answer>blue</answer><answer>green</answer>",
        ]
        for output in invalid:
            with self.subTest(output=output):
                self.assertFalse(extract_answer(output)[1])
        self.assertEqual(
            extract_answer("\nReasoning on multiple lines.\nVisible evidence supports green.\n"
                           "<answer> green </answer> \n"),
            ("green", True),
        )

    def test_malformed_terminal_answer_preserves_semantic_accuracy(self):
        response = MagicMock()
        response.__enter__.return_value = response
        with (
            patch.dict("os.environ", {"DEEPEYES_JUDGE_API_KEY": "test-key",
                                      "DEEPEYES_REPETITION_ZERO_REWARD": "false"}),
            patch("groove.deepeyes_reward.urllib.request.urlopen", return_value=response),
            patch("groove.deepeyes_reward.json.load",
                  return_value={"choices": [{"message": {"content": "1"}}]}),
        ):
            result = _judge_one("What color?", "green", "Reason. <answer>green</answer> trailing")
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["format_valid"], 0.0)
        self.assertAlmostEqual(result["score"], 0.8)

    def test_unconstrained_explanation_is_not_guessed(self):
        with self.assertRaises(ValueError):
            parse_judgement("The answers describe different objects.")

    def test_reward_module_supports_verl_external_object_loader(self):
        from verl.utils.import_utils import load_extern_object

        module_path = Path(__file__).parents[1] / "src/groove/deepeyes_reward.py"
        loaded = load_extern_object(module_path=str(module_path), object_name="compute_score")

        self.assertTrue(callable(loaded))

    def test_severe_repetition_zeroes_reward_but_preserves_accuracy(self):
        response = MagicMock()
        response.__enter__.return_value = response
        payload = {"choices": [{"message": {"content": "1"}}]}
        repeated_output = "<answer>" + ("blue " * 40).strip() + "</answer>"
        environment = {
            "DEEPEYES_JUDGE_API_KEY": "test-key",
            "DEEPEYES_JUDGE_MAX_RETRIES": "0",
            "DEEPEYES_REPETITION_ZERO_REWARD": "true",
        }
        with (
            patch.dict("os.environ", environment, clear=False),
            patch("groove.deepeyes_reward.urllib.request.urlopen", return_value=response),
            patch("groove.deepeyes_reward.json.load", return_value=payload),
        ):
            result = _judge_one("What color?", "blue", repeated_output)

        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["answer_reward"], 0.0)
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["format_reward"], 0.0)
        self.assertEqual(result["severe_repetition"], 1.0)
        self.assertEqual(result["repetition_zeroed_reward"], 1.0)

    def test_inner_repeat_detects_a_degenerate_suffix(self):
        varied_prefix = " ".join(f"distinct{index}" for index in range(160))
        repetitive_suffix = "looping answer token sequence " * 10

        hit = find_inner_repetition(f"{varied_prefix} {repetitive_suffix}")

        self.assertIsNotNone(hit)
        assert hit is not None
        self.assertGreaterEqual(hit.repeats, 4)
        self.assertGreaterEqual(hit.total_characters, 80)

    def test_distributed_restatements_are_not_a_contiguous_loop(self):
        repeated_idea = "The object is on the left."
        output = " ".join(
            f"{repeated_idea} Distinct observation number {index} changes the context."
            for index in range(6)
        )

        self.assertIsNone(find_inner_repetition(output))

    def test_short_repetition_does_not_trigger_the_default_gate(self):
        response = MagicMock()
        response.__enter__.return_value = response
        payload = {"choices": [{"message": {"content": "1"}}]}
        environment = {
            "DEEPEYES_JUDGE_API_KEY": "test-key",
            "DEEPEYES_JUDGE_MAX_RETRIES": "0",
            "DEEPEYES_REPETITION_ZERO_REWARD": "true",
        }
        with (
            patch.dict("os.environ", environment, clear=False),
            patch("groove.deepeyes_reward.urllib.request.urlopen", return_value=response),
            patch("groove.deepeyes_reward.json.load", return_value=payload),
        ):
            result = _judge_one("What color?", "blue", "<answer>blue blue blue</answer>")

        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["severe_repetition"], 0.0)


if __name__ == "__main__":
    unittest.main()
