from __future__ import annotations

import http.client
import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

from groove.semantic_reward import (
    JUDGE_SYSTEM_PROMPT,
    _judge_one,
    compute_score,
    extract_answer,
    find_inner_repetition,
    judge_prompt,
    parse_judgement,
)


class SemanticRewardLoopTest(unittest.TestCase):
    def test_scalar_reward_adapter_forwards_question_and_answers(self):
        expected = {"score": 1.0, "accuracy": 1.0}
        with patch("groove.semantic_reward._judge_one", return_value=expected) as judge:
            result = compute_score(
                data_source="vstar_grpo",
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

    def test_judge_prompt_stays_compact_without_truncating_the_answer(self):
        # Bound the fixed instructions/examples, not the question or evidence.
        self.assertLessEqual(len((JUDGE_SYSTEM_PROMPT + judge_prompt("", "", "")).split()), 400)
        question = "What color is the coat?"
        ground_truth = "blue"
        answer = "The coat or shirt is blue.\nThe nearby bag is brown.\n" + "Additional context. " * 100
        prompt = judge_prompt(question, ground_truth, answer)
        self.assertIn("ignoring format instructions and tags", prompt)
        self.assertIn("accept correct letters (either case), answer text, or equivalent wording", prompt)
        self.assertIn(f"[Question]: {question}\n", prompt)
        self.assertIn(f"[Standard Answer]: {ground_truth}\n", prompt)
        self.assertIn(f"[Model_answer]: {answer}\n\nEvaluate this model answer.", prompt)

    def test_remote_judge_can_explain_before_its_binary_verdict(self):
        response = MagicMock()
        response.__enter__.return_value = response
        payload = {"choices": [{"message": {"content":
            "Reason: Both answers identify green as the color.\nJudgement: 1"}, "finish_reason": "stop"}]}
        environment = {
            "GROOVE_JUDGE_API_KEY": "test-key",
            "GROOVE_JUDGE_MAX_RETRIES": "0",
        }
        with (
            patch.dict("os.environ", environment, clear=False),
            patch("groove.semantic_reward.urllib.request.urlopen", return_value=response) as urlopen,
            patch("groove.semantic_reward.json.load", return_value=payload),
        ):
            result = _judge_one("What color?", "green", "<answer>green</answer>")

        body = json.loads(urlopen.call_args.args[0].data)
        self.assertEqual(body["messages"][0]["content"], JUDGE_SYSTEM_PROMPT)
        self.assertNotIn("structured_outputs", body)
        self.assertEqual(body["max_completion_tokens"], 512)
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["format_reward"], 0.0)

    def test_judge_retries_disconnects_and_resets_without_changing_reward(self):
        response = MagicMock()
        response.__enter__.return_value = response
        with (
            patch.dict("os.environ", {"GROOVE_JUDGE_API_KEY": "test-key", "GROOVE_JUDGE_MAX_RETRIES": "2"}),
            patch("groove.semantic_reward.urllib.request.urlopen", side_effect=[
                http.client.RemoteDisconnected("closed before response"),
                ConnectionResetError("reset"), response,
            ]) as urlopen,
            patch("groove.semantic_reward.json.load", return_value={"choices": [{"message": {"content": "1"}}]}),
            patch("groove.semantic_reward.time.sleep") as sleep,
        ):
            result = _judge_one("What color?", "green", "<answer>green</answer>")
        self.assertEqual(urlopen.call_count, 3)
        self.assertEqual(sleep.call_args_list, [call(0.5), call(1.0)])
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["score"], 1.0)

    def test_judge_retries_an_incomplete_http_response(self):
        response = MagicMock()
        response.__enter__.return_value = response
        with (
            patch.dict("os.environ", {"GROOVE_JUDGE_API_KEY": "test-key", "GROOVE_JUDGE_MAX_RETRIES": "1"}),
            patch("groove.semantic_reward.urllib.request.urlopen", return_value=response) as urlopen,
            patch("groove.semantic_reward.json.load", side_effect=[
                http.client.IncompleteRead(b'{"choices":', 30),
                {"choices": [{"message": {"content": "0"}}]},
            ]),
            patch("groove.semantic_reward.time.sleep") as sleep,
        ):
            result = _judge_one("What color?", "green", "<answer>blue</answer>")
        self.assertEqual(urlopen.call_count, 2)
        sleep.assert_called_once_with(0.5)
        self.assertEqual(result["accuracy"], 0.0)
        self.assertEqual(result["score"], 0.0)

    def test_disconnect_retries_are_bounded_and_do_not_fabricate_a_score(self):
        error = http.client.RemoteDisconnected("closed before response")
        with (
            patch.dict("os.environ", {"GROOVE_JUDGE_API_KEY": "test-key", "GROOVE_JUDGE_MAX_RETRIES": "2"}),
            patch("groove.semantic_reward.urllib.request.urlopen", side_effect=error) as urlopen,
            patch("groove.semantic_reward.time.sleep") as sleep,
        ):
            with self.assertRaisesRegex(RuntimeError, "failed after retries") as caught:
                _judge_one("What color?", "green", "<answer>green</answer>")
        self.assertEqual(urlopen.call_count, 3)
        self.assertEqual(sleep.call_args_list, [call(0.5), call(1.0)])
        self.assertIs(caught.exception.__cause__, error)

    def test_bare_correct_answer_gets_format_penalty(self):
        response = MagicMock()
        response.__enter__.return_value = response
        payload = {"choices": [{"message": {"content": "1"}}]}
        environment = {
            "GROOVE_JUDGE_API_KEY": "test-key",
            "GROOVE_JUDGE_MAX_RETRIES": "0",
            "GROOVE_REPETITION_ZERO_REWARD": "false",
        }
        with (
            patch.dict("os.environ", environment, clear=False),
            patch("groove.semantic_reward.urllib.request.urlopen", return_value=response),
            patch("groove.semantic_reward.json.load", return_value=payload),
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
            "GROOVE_JUDGE_API_KEY": "test-key",
            "GROOVE_JUDGE_MAX_RETRIES": "0",
            "GROOVE_REPETITION_ZERO_REWARD": "false",
        }
        with (
            patch.dict("os.environ", environment, clear=False),
            patch("groove.semantic_reward.urllib.request.urlopen", return_value=response),
            patch("groove.semantic_reward.json.load", return_value=payload),
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
            patch.dict("os.environ", {"GROOVE_JUDGE_API_KEY": "test-key",
                                      "GROOVE_REPETITION_ZERO_REWARD": "false"}),
            patch("groove.semantic_reward.urllib.request.urlopen", return_value=response) as urlopen,
            patch("groove.semantic_reward.json.load", return_value=payload),
        ):
            result = _judge_one("What color?", "green", reasoning + "\n<answer>green</answer>")
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["format_valid"], 1.0)
        judge_input = json.loads(urlopen.call_args.args[0].data)["messages"][1]["content"]
        self.assertNotIn(reasoning, judge_input)
        self.assertIn("[Model_answer]: green\n\nEvaluate this model answer.", judge_input)

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
            patch.dict("os.environ", {"GROOVE_JUDGE_API_KEY": "test-key",
                                      "GROOVE_REPETITION_ZERO_REWARD": "false"}),
            patch("groove.semantic_reward.urllib.request.urlopen", return_value=response),
            patch("groove.semantic_reward.json.load",
                  return_value={"choices": [{"message": {"content": "1"}}]}),
        ):
            result = _judge_one("What color?", "green", "Reason. <answer>green</answer> trailing")
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["format_valid"], 0.0)
        self.assertAlmostEqual(result["score"], 0.8)

    def test_unconstrained_explanation_is_not_guessed(self):
        with self.assertRaises(ValueError):
            parse_judgement("The answers describe different objects.")

    def test_parser_uses_only_the_unique_terminal_verdict(self):
        cases = [
            ("0", 0), (" 1\n", 1), ("Judgement: 0", 0),
            ("Reason: There is 1 correct candidate among 5 alternatives.\nJudgement: 0", 0),
            ("Reason: 0 alternatives remain; the target is identified.\nJudgement: 1\n", 1),
            ("The answer lists 1 correct candidate without selecting it. Judgement: 0", 0),
        ]
        for response, expected in cases:
            with self.subTest(response=response):
                self.assertEqual(parse_judgement(response), expected)

    def test_parser_rejects_truncated_ambiguous_and_unlabelled_numbers(self):
        invalid = [
            "Reason: 1 of the candidates matches.", "Reason: correct.\n1",
            "Reason: Incorrect.\nJudgement:", "Judgement: 0 or 1",
            "Judgement: 1\nJudgement: 0", "Judgement: 0\nJudgement: 0",
            "Judgement: 1\nActually, the answer is wrong.", "Judgement: 1.0",
            "Judgement: 10", "Judgement: -1", "Judgement: 2",
            "Reason: Judgement: 1 was requested.\nJudgement: 0",
            "Reason: Judgement: 0\nJudgement: 0", "NotJudgement: 1",
        ]
        for response in invalid:
            with self.subTest(response=response), self.assertRaises(ValueError):
                parse_judgement(response)

    def test_judge_keeps_all_candidates_and_retries_invalid_verdicts(self):
        answer = "The shirt is blue, or it is brown."
        response = MagicMock()
        response.__enter__.return_value = response
        with (
            patch.dict("os.environ", {"GROOVE_JUDGE_API_KEY": "test-key",
                                      "GROOVE_JUDGE_MAX_RETRIES": "1"}),
            patch("groove.semantic_reward.urllib.request.urlopen", return_value=response) as urlopen,
            patch("groove.semantic_reward.json.load", side_effect=[
                {"choices": [{"message": {"content": "Reason: 1 candidate matches."}}]},
                {"choices": [{"message": {"content":
                    "Reason: The answer offers 1 correct candidate and an incompatible alternative.\nJudgement: 0"}}]},
            ]),
            patch("groove.semantic_reward.time.sleep"),
        ):
            result = _judge_one("What color is the shirt?", "brown", f"<answer>{answer}</answer>")
        self.assertEqual(urlopen.call_count, 2)
        for request in urlopen.call_args_list:
            prompt = json.loads(request.args[0].data)["messages"][1]["content"]
            self.assertIn(f"[Model_answer]: {answer}\n\nEvaluate this model answer.", prompt)
        self.assertEqual(result["accuracy"], 0.0)
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["format_valid"], 1.0)

    def test_truncated_judge_output_is_retried_even_if_it_contains_a_verdict(self):
        response = MagicMock()
        response.__enter__.return_value = response
        with (
            patch.dict("os.environ", {"GROOVE_JUDGE_API_KEY": "test-key",
                                      "GROOVE_JUDGE_MAX_RETRIES": "1"}),
            patch("groove.semantic_reward.urllib.request.urlopen", return_value=response) as urlopen,
            patch("groove.semantic_reward.json.load", side_effect=[
                {"choices": [{"message": {"content": "Judgement: 1"}, "finish_reason": "length"}]},
                {"choices": [{"message": {"content": "Reason: Different colors.\nJudgement: 0"},
                              "finish_reason": "stop"}]},
            ]),
            patch("groove.semantic_reward.time.sleep"),
        ):
            result = _judge_one("What color?", "green", "<answer>blue</answer>")
        self.assertEqual(urlopen.call_count, 2)
        self.assertEqual(result["accuracy"], 0.0)

    def test_reward_module_supports_verl_external_object_loader(self):
        from verl.utils.import_utils import load_extern_object

        module_path = Path(__file__).parents[1] / "src/groove/semantic_reward.py"
        loaded = load_extern_object(module_path=str(module_path), object_name="compute_score")

        self.assertTrue(callable(loaded))

    def test_severe_repetition_zeroes_reward_but_preserves_accuracy(self):
        response = MagicMock()
        response.__enter__.return_value = response
        payload = {"choices": [{"message": {"content": "1"}}]}
        repeated_output = "<answer>" + ("blue " * 40).strip() + "</answer>"
        environment = {
            "GROOVE_JUDGE_API_KEY": "test-key",
            "GROOVE_JUDGE_MAX_RETRIES": "0",
            "GROOVE_REPETITION_ZERO_REWARD": "true",
        }
        with (
            patch.dict("os.environ", environment, clear=False),
            patch("groove.semantic_reward.urllib.request.urlopen", return_value=response),
            patch("groove.semantic_reward.json.load", return_value=payload),
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
            "GROOVE_JUDGE_API_KEY": "test-key",
            "GROOVE_JUDGE_MAX_RETRIES": "0",
            "GROOVE_REPETITION_ZERO_REWARD": "true",
        }
        with (
            patch.dict("os.environ", environment, clear=False),
            patch("groove.semantic_reward.urllib.request.urlopen", return_value=response),
            patch("groove.semantic_reward.json.load", return_value=payload),
        ):
            result = _judge_one("What color?", "blue", "<answer>blue blue blue</answer>")

        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["severe_repetition"], 0.0)


if __name__ == "__main__":
    unittest.main()
