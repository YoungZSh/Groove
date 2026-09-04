from __future__ import annotations

import unittest
from unittest.mock import patch

from groove.deepeyes_reward import compute_score


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


if __name__ == "__main__":
    unittest.main()
