from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch

from groove.deepeyes_reward import (
    _judge_one,
    compute_score,
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
            result = _judge_one("What color?", "green", "green")

        body = json.loads(urlopen.call_args.args[0].data)
        self.assertEqual(body["structured_outputs"], {"choice": ["0", "1"]})
        self.assertEqual(body["max_completion_tokens"], 4)
        self.assertEqual(result["score"], 1.0)

    def test_unconstrained_explanation_is_not_guessed(self):
        with self.assertRaises(ValueError):
            parse_judgement("The answers describe different objects.")

    def test_severe_repetition_zeroes_reward_but_preserves_accuracy(self):
        response = MagicMock()
        response.__enter__.return_value = response
        payload = {"choices": [{"message": {"content": "1"}}]}
        repeated_output = ("The answer is blue. <answer>blue</answer> " * 40).strip()
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
            result = _judge_one("What color?", "blue", "blue blue blue <answer>blue</answer>")

        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["severe_repetition"], 0.0)


if __name__ == "__main__":
    unittest.main()
