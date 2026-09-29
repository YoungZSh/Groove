from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image
import pyarrow.parquet as pq

spec = spec_from_file_location("prepare_opd", Path(__file__).parents[1] / "scripts/prepare_vision_opd_grpo.py")
preparation = module_from_spec(spec)
spec.loader.exec_module(preparation)


class VisionOpdPreparationTest(unittest.TestCase):
    def row(self, index=0):
        question = "What color is the stage floor lighting?\n\nA. black\nB. white\nC. blue\nD. purple\n\nAnswer with the option's letter from the given choices."
        return {"original_images": [f"original_images/{index}.png"],
                "images": ["images/PRIVATE_OVERLAY.png"],
                "teacher_images": ["teacher_images/PRIVATE_CROP.png"], "bbox": [1, 2, 3, 4],
                "problem": "<image>\n" + preparation.BOX_HINT + "\n" + question,
                "answer": "D", "extra_info": {"question": question, "answer": "D"}}

    def source(self, folder, count=1):
        root = Path(folder)
        (root / "original_images").mkdir()
        rows = [self.row(i) for i in range(count)]
        for i in range(count):
            Image.new("RGB", (32, 32)).save(root / f"original_images/{i}.png")
        source = root / "train.jsonl"
        source.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return source, rows

    def test_unboxed_inputs_current_format_and_full_semantic_reference(self):
        with TemporaryDirectory() as folder:
            source, rows = self.source(folder)
            before = deepcopy(rows[0])
            result = preparation.build_record(rows[0], source.parent, 0, 53)
            self.assertEqual(rows[0], before)
            self.assertEqual(result["images"], [{"path": str(source.parent / "original_images/0.png")}])
            self.assertEqual(result["reward_model"]["ground_truth"], "(D) purple")
            self.assertEqual(result["extra_info"]["question_id"], "vision-opd-000053")
            self.assertIn("<answer>...</answer>", result["prompt"][1]["content"])
            for forbidden in ("red bounding box", "FINAL:", "PRIVATE_", "bbox", "teacher_images"):
                self.assertNotIn(forbidden, json.dumps(result))
            self.assertNotIn("Answer with", result["extra_info"]["question"])
            rows[0]["extra_info"].pop("question")
            self.assertEqual(preparation.build_record(rows[0], source.parent, 0, 53), result)

    def test_missing_original_overlay_path_and_conflicting_label_are_rejected(self):
        with TemporaryDirectory() as folder:
            source, rows = self.source(folder)
            for field, value in (("original_images", ["images/PRIVATE_OVERLAY.png"]),
                                 ("original_images", ["original_images/missing.png"]),
                                 ("original_images", ["original_images/../images/overlay.png"]),
                                 ("answer", "A")):
                item = deepcopy(rows[0]); item[field] = value
                with self.assertRaises(ValueError):
                    preparation.build_record(item, source.parent, 0, 0)

    def test_audit_hash_is_required_and_only_explicit_categories_are_excluded(self):
        with TemporaryDirectory() as folder:
            source, rows = self.source(folder, 4)
            audit = source.parent / "audit.json"
            records = [{"filtered_row": i, "original_image": row["original_images"][0],
                        "review_category": "confirmed_defect" if i == 0 else "needs_review",
                        "unboxed_use": "rewrite_required" if i == 1 else "rewrite_recommended",
                        "audit_id": str(i), "audit_notes": "audit only"} for i, row in enumerate(rows)]
            payload = {"method": {"dataset_sha256": preparation.digest(source)}, "records": records}
            audit.write_text(json.dumps(payload))
            self.assertEqual(set(preparation.audit_exclusions(audit, preparation.digest(source), rows)), {0, 1})
            with self.assertRaisesRegex(ValueError, "hash"):
                preparation.audit_exclusions(audit, "wrong", rows)
            output = source.parent / "prepared"
            original_hash = preparation.digest(source)
            manifest = preparation.prepare(source, output, audit)
            prepared = pq.read_table(output / "train.parquet").to_pylist()
            self.assertEqual([r["extra_info"]["source_index"] for r in prepared], [2, 3])
            self.assertEqual(manifest["excluded_rows"], 2)
            self.assertEqual(preparation.digest(source), original_hash)
            self.assertIn("PRIVATE_CROP", (output / "lineage.jsonl").read_text())
            self.assertNotIn("PRIVATE_CROP", json.dumps(prepared))
            with self.assertRaises(FileExistsError):
                preparation.prepare(source, output, audit)

    def test_explicit_annotation_reference_is_quarantined_without_rewriting_answer(self):
        with TemporaryDirectory() as folder:
            source, rows = self.source(folder, 2)
            rows[1]["extra_info"]["question"] = rows[1]["extra_info"]["question"].replace(
                "stage floor lighting", "highlighted rectangular area")
            source.write_text("".join(json.dumps(row) + "\n" for row in rows))
            output = source.parent / "prepared"
            manifest = preparation.prepare(source, output)
            self.assertEqual(manifest["rows"], 1)
            self.assertEqual(manifest["exclusion_reason_counts"], {"explicit_annotation_reference": 1})

    def test_random_sample_is_reproducible_unique_and_keeps_lineage_aligned(self):
        with TemporaryDirectory() as folder:
            source, rows = self.source(folder, 10)
            first = preparation.prepare(source, source.parent / "first", sample_size=5, seed=20260904)
            second = preparation.prepare(source, source.parent / "second", sample_size=5, seed=20260904)
            third = preparation.prepare(source, source.parent / "third", sample_size=5, seed=20260905)
            self.assertEqual(first["selected_source_indices"], second["selected_source_indices"])
            self.assertNotEqual(first["selected_source_indices"], third["selected_source_indices"])
            self.assertEqual(len(set(first["selected_source_indices"])), 5)
            lineage = [json.loads(line) for line in (source.parent / "first/lineage.jsonl").read_text().splitlines()]
            self.assertEqual([row["source_index"] for row in lineage], first["selected_source_indices"])
            for count in (0, 11):
                with self.assertRaises(ValueError):
                    preparation.prepare(source, source.parent / "invalid", sample_size=count)
                self.assertFalse((source.parent / "invalid").exists())


if __name__ == "__main__":
    unittest.main()
