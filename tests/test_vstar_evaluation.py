from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import unittest


spec = spec_from_file_location("vstar_evaluation", Path(__file__).parents[1] / "scripts/evaluate_vstar.py")
evaluation = module_from_spec(spec)
spec.loader.exec_module(evaluation)


class VStarScorerTest(unittest.TestCase):
    choices = {"A": "rubber", "B": "cotton", "C": "kevlar", "D": "leather"}

    def test_supported_answers(self):
        for text, expected in [("<answer>A</answer>", "A"), ("<answer>(D) leather</answer>", "D"),
                               ("A", "A"), ("<answer>cotton</answer>", "B"),
                               ("Think about B and C.\n<answer>A</answer>", "A"),
                               ("Consider C.\nFinal answer: D", "D")]:
            with self.subTest(text=text):
                self.assertEqual(evaluation.parse_prediction(text, self.choices)["predicted_label"], expected)

    def test_does_not_guess_from_ambiguous_reasoning(self):
        for text in ["A or B", "I considered (A), (B), (C) and (D).", "<answer>A leather</answer>",
                     "<answer>E</answer>", "<answer></answer>", "<answer>A or B</answer>"]:
            with self.subTest(text=text):
                self.assertIsNone(evaluation.parse_prediction(text, self.choices)["predicted_label"])

    def test_format_diagnostic_does_not_change_semantic_label(self):
        result = evaluation.parse_prediction("<answer>A</answer><answer>D</answer>", self.choices)
        self.assertEqual(result["predicted_label"], "D")
        self.assertFalse(result["format_valid"])
        self.assertFalse(evaluation.parse_prediction("A", self.choices)["format_valid"])
        self.assertTrue(evaluation.parse_prediction("Reasoning. <answer>A</answer>\n", self.choices)["format_valid"])

    def test_source_question_preserved_except_output_instruction(self):
        source = "Question?\n(A) rubber\n(B) cotton\nAnswer with the option's letter from the given choices directly."
        question = evaluation.question_text(source)
        self.assertTrue(question.startswith("Question?\n(A) rubber\n(B) cotton\n"))
        self.assertNotIn("directly.", question)
        self.assertIn("<answer>", question)


if __name__ == "__main__":
    unittest.main()
