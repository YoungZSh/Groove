from __future__ import annotations

from copy import deepcopy
import hashlib
import http.client
import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, call, patch

import pyarrow as pa
import pyarrow.parquet as pq

from groove.response_prompt import REASONING_SYSTEM_PROMPT
from groove.semantic_reward import compute_score, compute_score_batched, extract_answer
from groove.vstar_bench import (
    LETTER_ANSWER_INSTRUCTION,
    SEMANTIC_ANSWER_INSTRUCTION,
    compute_validation_score,
    question_text,
    question_without_response_format,
)


spec = spec_from_file_location("prepare_validation", Path(__file__).parents[1] / "scripts/prepare_vstar_validation.py")
preparation = module_from_spec(spec)
spec.loader.exec_module(preparation)


class VStarValidationTest(unittest.TestCase):
    choices = {"A": "rubber", "B": "cotton", "C": "kevlar", "D": "leather"}

    def source_rows(self):
        return [{
            "question_id": str(index),
            "image": {"bytes": b"original-image-payload", "path": "archived.jpg"},
            "text": "What material?\n(A) rubber\n(B) cotton\n(C) kevlar\n(D) leather\n"
                    "Answer with the option's letter from the given choices directly.",
            "label": "D",
            "category": "direct_attributes" if index < 115 else "relative_position",
        } for index in range(191)]

    def info(self):
        return {"split": "validation", "choices": self.choices, "question": "What material?"}

    def test_preparation_preserves_all_questions_original_images_and_order(self):
        source = self.source_rows()
        original = deepcopy(source)
        records = preparation.build_records(source)
        self.assertEqual(source, original)
        self.assertEqual([row["extra_info"]["question_id"] for row in records], [str(i) for i in range(191)])
        for row in records:
            self.assertEqual(row["data_source"], "vstar_bench")
            self.assertEqual(row["images"], [{"bytes": b"original-image-payload"}])
            self.assertEqual(row["reward_model"]["ground_truth"], "D")
            self.assertEqual(row["prompt"][0]["content"], REASONING_SYSTEM_PROMPT)
            user_message = row["prompt"][1]["content"]
            self.assertIn("(A) rubber\n(B) cotton\n(C) kevlar\n(D) leather", user_message)
            self.assertNotIn("FINAL:", user_message)
            self.assertNotIn("directly.", user_message)
            self.assertIn("<answer>...</answer>", user_message)
            self.assertIn(SEMANTIC_ANSWER_INSTRUCTION, user_message)
            self.assertEqual(row["extra_info"]["question"],
                             "What material?\n(A) rubber\n(B) cotton\n(C) kevlar\n(D) leather")
            self.assertEqual(row["extra_info"]["choices"], self.choices)
            self.assertEqual(row["extra_info"]["split"], "validation")

    def test_incomplete_duplicate_or_corrupt_source_is_rejected(self):
        partial = self.source_rows()[:-1]
        duplicate = self.source_rows()
        duplicate[-1]["question_id"] = "0"
        wrong_category = self.source_rows()
        wrong_category[0]["category"] = "relative_position"
        missing_image = self.source_rows()
        missing_image[0]["image"]["bytes"] = None
        wrong_label = self.source_rows()
        wrong_label[0]["label"] = "E"
        for name, rows in (("partial", partial), ("duplicate", duplicate),
                           ("category", wrong_category), ("image", missing_image), ("label", wrong_label)):
            with self.subTest(case=name), self.assertRaises(ValueError):
                preparation.build_records(rows)

    def test_prepared_file_and_manifest_are_reproducible_and_never_overwritten(self):
        with TemporaryDirectory() as folder:
            source = Path(folder) / "source.parquet"
            output = Path(folder) / "validation.parquet"
            pq.write_table(pa.Table.from_pylist(self.source_rows()), source)
            manifest = preparation.prepare_validation(source, output)
            self.assertEqual(pq.ParquetFile(output).metadata.num_rows, 191)
            self.assertEqual(manifest["source_sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
            before = output.read_bytes()
            self.assertEqual(manifest["validation_sha256"], hashlib.sha256(before).hexdigest())
            with self.assertRaises(FileExistsError):
                preparation.prepare_validation(source, output)
            self.assertEqual(output.read_bytes(), before)

    def judge(self, output, verdict="1", info=None):
        response = MagicMock()
        response.__enter__.return_value = response
        environment = {"GROOVE_JUDGE_API_KEY": "test-key", "GROOVE_JUDGE_MAX_RETRIES": "0",
                       "GROOVE_REPETITION_ZERO_REWARD": "true"}
        with patch.dict("os.environ", environment), \
                patch("groove.semantic_reward.urllib.request.urlopen", return_value=response) as request, \
                patch("groove.semantic_reward.json.load", return_value={"choices": [{"message": {"content": verdict}}]}):
            result = compute_score("vstar_bench", output, "D", info or self.info(),
                                   answer_reward_weight=0.3, format_reward_weight=0.9)
        self.assertEqual(request.call_count, 1)
        return result, json.loads(request.call_args.args[0].data)

    def test_every_benchmark_answer_uses_training_extraction_and_semantic_judge(self):
        cases = [("<answer>D</answer>", "1"), ("D", "1"),
                 ("Reasoning outside. <answer>The glove is made of leather.</answer>", "1"),
                 ("<answer>A</answer>", "0"), ("A or D", "0"),
                 ("<ANSWER>leather</ANSWER>", "1"),
                 ("<answer>cotton</answer><answer>leather</answer>", "1")]
        for output, verdict in cases:
            with self.subTest(output=output):
                result, body = self.judge(output, verdict)
                answer, valid = extract_answer(output)
                self.assertEqual(result["score"], float(verdict))
                self.assertEqual(result["accuracy"], float(verdict))
                self.assertEqual(result["format_valid"], float(valid))
                self.assertEqual(result["format_reward_weight"], 0.0)
                self.assertEqual(result["answer_reward_weight"], 1.0)
                self.assertNotIn("structured_outputs", body)
                self.assertEqual(body["max_completion_tokens"], 512)
                prompt = body["messages"][1]["content"]
                self.assertIn("[Standard Answer]: (D) leather\n", prompt)
                self.assertIn(f"[Model_answer]: {answer}\n\nEvaluate this model answer.", prompt)
                self.assertIn("(A) rubber\n(B) cotton\n(C) kevlar\n(D) leather", prompt)
                self.assertNotIn("Reasoning outside.", prompt)

    def test_rule_unparsed_natural_language_can_receive_full_semantic_accuracy(self):
        result, _ = self.judge("<answer>The glove is made of leather.</answer>")
        self.assertEqual(result["rule_unparsed"], 1.0)
        self.assertEqual(result["rule_accuracy"], 0.0)
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["semantic_judge"], 1.0)
        self.assertNotIn("unparsed", result)

    def test_question_formatting_preserves_options_and_the_legacy_evaluator_protocol(self):
        # An instruction-like option is answer content, not a suffix to remove.
        question = "What does the sign say?\n(A) Return the selected option letter.\n(B) Stop."
        for suffix in (LETTER_ANSWER_INSTRUCTION, SEMANTIC_ANSWER_INSTRUCTION,
                       "Answer with the option's letter from the given choices directly.",
                       "Answer with the option letter directly.", "Return the selected option letter."):
            with self.subTest(suffix=suffix):
                source = question + "\n" + suffix + "\n \t"
                self.assertEqual(question_without_response_format(source), question)
                self.assertEqual(question_text(source), question + "\n" + LETTER_ANSWER_INSTRUCTION)
                adapted = question_text(source, allow_answer_text=True)
                self.assertEqual(adapted, question + "\n" + SEMANTIC_ANSWER_INSTRUCTION)
                self.assertEqual(question_text(adapted, allow_answer_text=True), adapted)
        self.assertEqual(question_without_response_format(question), question)

    def test_validation_removes_student_format_instructions_from_real_judge_requests(self):
        question = "What material?\n(A) rubber\n(B) cotton\n(C) kevlar\n(D) leather"
        for suffix in (LETTER_ANSWER_INSTRUCTION, SEMANTIC_ANSWER_INSTRUCTION,
                       "Answer with the option's letter from the given choices directly."):
            info = {**self.info(), "question": question + "\n" + suffix}
            original = deepcopy(info)
            for answer in ("D", "d", "leather", "The glove is made of leather.", "A leather glove."):
                with self.subTest(suffix=suffix, answer=answer):
                    result, body = self.judge(f"<answer>{answer}</answer>", info=info)
                    prompt = body["messages"][1]["content"]
                    self.assertIn(f"[Question]: {question}\n[Standard Answer]: (D) leather\n", prompt)
                    self.assertIn(f"[Model_answer]: {answer}\n", prompt)
                    self.assertNotIn(suffix, prompt)
                    self.assertEqual(result["score"], 1.0)
                    self.assertEqual(info, original)

    def test_historical_purple_case_sends_both_reference_letter_and_answer_text(self):
        question = "What is the color of the comb?\n(A) brown\n(B) red\n(C) purple\n(D) black"
        info = {"question": question + "\n" + LETTER_ANSWER_INSTRUCTION, "split": "validation",
                "choices": {"A": "brown", "B": "red", "C": "purple", "D": "black"}}
        with patch("groove.semantic_reward._judge_one", return_value={"score": 1.0, "accuracy": 1.0}) as judge:
            result = compute_score("vstar_bench", "<answer>purple</answer>", "C", info)
        judge.assert_called_once_with(question, "(C) purple", "<answer>purple</answer>",
                                      answer_reward_weight=1.0, format_reward_weight=0.0,
                                      apply_training_shaping=False)
        self.assertEqual(result["score"], 1.0)

    def test_validation_preserves_conflicting_letter_text_and_unresolved_candidates(self):
        # Never normalize a contradictory answer into its matching word/letter.
        for answer in ("(D) rubber", "Option A: leather", "D or A"):
            with self.subTest(answer=answer):
                result, body = self.judge(f"<answer>{answer}</answer>", verdict="0")
                self.assertIn(f"[Model_answer]: {answer}\n", body["messages"][1]["content"])
                self.assertEqual(result["accuracy"], 0.0)

    def test_validation_reads_explained_verdict_without_training_shaping(self):
        result, _ = self.judge("D", "Reason: D selects the reference material.\nJudgement: 1")
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["format_valid"], 0.0)
        self.assertEqual(result["format_reward_weight"], 0.0)

    def test_validation_repetition_is_diagnostic_and_does_not_zero_judge_accuracy(self):
        output = "repeated reasoning phrase " * 10 + "<answer>leather</answer>"
        result, _ = self.judge(output)
        self.assertEqual(result["severe_repetition"], 1.0)
        self.assertEqual(result["repetition_zeroed_reward"], 0.0)
        self.assertEqual(result["answer_reward"], 1.0)
        self.assertEqual(result["score"], 1.0)

    def test_semantic_validation_includes_real_options_once_and_never_sends_null_options(self):
        info = {**self.info(), "choices": {"A": "rubber", "B": None, "C": None, "D": "leather"},
                "question": "What material?\n(A) rubber\n(D) leather\nReturn the selected option letter."}
        before = deepcopy(info)
        _, body = self.judge("<answer>D</answer>", info=info)
        prompt = body["messages"][1]["content"]
        self.assertEqual(prompt.count("(A) rubber"), 1)
        self.assertNotIn("(B) None", prompt)
        self.assertNotIn("Return the selected option letter.", prompt)
        self.assertEqual(info, before)

    def test_validation_judge_failure_is_not_silently_converted_to_rule_score(self):
        with patch.dict("os.environ", {"GROOVE_JUDGE_API_KEY": "test-key", "GROOVE_JUDGE_MAX_RETRIES": "1"}), \
                patch("groove.semantic_reward.urllib.request.urlopen",
                      side_effect=http.client.RemoteDisconnected("closed")) as request, \
                patch("groove.semantic_reward.time.sleep"):
            with self.assertRaisesRegex(RuntimeError, "failed after retries"):
                compute_score("vstar_bench", "<answer>D</answer>", "D", self.info())
        self.assertEqual(request.call_count, 2)

    def test_training_reward_and_mixed_batch_order_are_preserved(self):
        expected = {"score": 0.8, "accuracy": 1.0}
        def score(question, reference, output, **kwargs):
            if not kwargs.get("apply_training_shaping", True):
                value = float(output == "<answer>D</answer>")
                return {"score": value, "accuracy": value}
            return dict(expected)

        with patch("groove.semantic_reward._judge_one", side_effect=score) as judge:
            result = compute_score_batched(
                ["vstar_bench", "visual_qa", "vstar_bench"],
                ["<answer>D</answer>", "rubber", "<answer>A</answer>"],
                ["D", "rubber", "D"],
                [self.info(), {"question": "What material?"}, self.info()],
            )
            self.assertEqual([r["score"] for r in result], [1.0, 0.8, 0.0])
            self.assertEqual(judge.call_count, 3)
            self.assertIn(call("What material?", "rubber", "rubber",
                               answer_reward_weight=1.0, format_reward_weight=0.2), judge.call_args_list)

    def test_parquet_null_options_are_not_answer_candidates(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "mixed-options.parquet"
            pq.write_table(pa.Table.from_pylist([
                {"choices": {"A": "rubber", "B": "leather"}},
                {"choices": self.choices},
            ]), path)
            choices = pq.read_table(path).to_pylist()[0]["choices"]
        self.assertIsNone(choices["C"])
        self.assertIsNone(choices["D"])
        info = {**self.info(), "choices": choices}
        before = deepcopy(info)
        for output, expected, unparsed in (
            ("<answer>B</answer>", 1.0, 0.0),
            ("<answer>B leather</answer>", 1.0, 0.0),
            ("<answer>leather</answer>", 1.0, 0.0),
            ("<answer>C</answer>", 0.0, 1.0),
            ("<answer>C cotton</answer>", 0.0, 1.0),
            ("<answer>unknown</answer>", 0.0, 1.0),
            ("<answer>None</answer>", 0.0, 1.0),
            ("<answer></answer>", 0.0, 1.0),
        ):
            with self.subTest(output=output):
                result = compute_validation_score(output, "B", info)
                self.assertEqual(result["score"], expected)
                self.assertEqual(result["unparsed"], unparsed)
        self.assertEqual(info, before)

    def test_benchmark_metadata_requires_validation_split_and_valid_reference(self):
        with self.assertRaises(ValueError):
            compute_validation_score("D", "D", {**self.info(), "split": "train"})
        with self.assertRaises(ValueError):
            compute_validation_score("D", "E", self.info())
        for missing in (None, "", " "):
            with self.subTest(reference=missing), self.assertRaises(ValueError):
                compute_validation_score("D", "D", {**self.info(), "choices": {"A": "rubber", "D": missing}})
        with patch("groove.semantic_reward._judge_one") as judge:
            for info in ({**self.info(), "question": ""}, {**self.info(), "split": "train"},
                         {**self.info(), "question": LETTER_ANSWER_INSTRUCTION}):
                with self.assertRaises(ValueError):
                    compute_score("vstar_bench", "D", "D", info)
            judge.assert_not_called()


if __name__ == "__main__":
    unittest.main()
