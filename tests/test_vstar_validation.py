from __future__ import annotations

from copy import deepcopy
import hashlib
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from groove.response_prompt import REASONING_SYSTEM_PROMPT
from groove.semantic_reward import compute_score, compute_score_batched
from groove.vstar_bench import compute_validation_score


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

    def test_benchmark_scoring_uses_no_remote_judge_and_no_format_penalty(self):
        cases = (("<answer>D</answer>", 1, 1), ("D", 1, 0),
                 ("<answer>leather</answer>", 1, 1), ("<answer>A</answer>", 0, 1),
                 ("A or D", 0, 0))
        with patch("groove.semantic_reward._judge_one") as judge:
            for output, accuracy, format_valid in cases:
                with self.subTest(output=output):
                    result = compute_score("vstar_bench", output, "D", self.info(),
                                           answer_reward_weight=0.9, format_reward_weight=0.2)
                    self.assertEqual(result["score"], accuracy)
                    self.assertEqual(result["accuracy"], accuracy)
                    self.assertEqual(result["format_valid"], format_valid)
            judge.assert_not_called()

    def test_training_reward_and_mixed_batch_order_are_preserved(self):
        expected = {"score": 0.8, "accuracy": 1.0}
        with patch("groove.semantic_reward._judge_one", return_value=expected) as judge:
            result = compute_score_batched(
                ["vstar_bench", "visual_qa", "vstar_bench"],
                ["<answer>D</answer>", "rubber", "<answer>A</answer>"],
                ["D", "rubber", "D"],
                [self.info(), {"question": "What material?"}, self.info()],
            )
            self.assertEqual([r["score"] for r in result], [1.0, 0.8, 0.0])
            judge.assert_called_once_with("What material?", "rubber", "rubber",
                                          answer_reward_weight=1.0, format_reward_weight=0.2)

    def test_benchmark_metadata_requires_validation_split_and_valid_reference(self):
        with self.assertRaises(ValueError):
            compute_validation_score("D", "D", {**self.info(), "split": "train"})
        with self.assertRaises(ValueError):
            compute_validation_score("D", "E", self.info())


if __name__ == "__main__":
    unittest.main()
