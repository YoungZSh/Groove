import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


class ValidationRolloutDumpTest(unittest.TestCase):
    def test_native_and_opsd_dump_every_validation_answer_and_drain_final_writes(self):
        from groove.verl_trainer import GrooveRayPPOTrainer
        from verl.trainer.ppo.v1 import PPOTrainerSync

        for trainer_type in (PPOTrainerSync, GrooveRayPPOTrainer):
            with self.subTest(trainer=trainer_type.__name__), TemporaryDirectory() as folder:
                trainer = trainer_type.__new__(trainer_type)
                trainer._init_dump_executor()
                root = Path(folder)
                training = root / "rollouts/10.jsonl"
                training.parent.mkdir()
                training.write_text("existing training rollout\n")
                outputs = [f"Reasoning for question {i}.\n<answer>The object is on the left.</answer>"
                           for i in range(191)]
                scores = [float(i % 2) for i in range(191)]
                try:
                    for step in (0, 10):
                        trainer.global_steps = step
                        trainer._dump_generations(
                            inputs=[f"Question {i}\n(A) left\n(B) right" for i in range(191)],
                            outputs=outputs, gts=["A"] * 191, scores=scores,
                            reward_extra_infos_dict={"accuracy": scores, "format_valid": [1.0] * 191,
                                                     "rule_unparsed": [1.0] * 191,
                                                     "semantic_judge": [1.0] * 191},
                            dump_path=str(root / "validation"),
                        )
                finally:
                    trainer._shutdown_dump_executor()
                for step in (0, 10):
                    rows = [json.loads(line) for line in (root / f"validation/{step}.jsonl").read_text().splitlines()]
                    self.assertEqual(len(rows), 191)
                    self.assertEqual([row["output"] for row in rows], outputs)
                    self.assertEqual([row["score"] for row in rows], scores)
                    self.assertEqual({row["step"] for row in rows}, {step})
                    self.assertTrue(all(row["rule_unparsed"] == 1.0 and row["semantic_judge"] == 1.0 for row in rows))
                    self.assertTrue(all(row["gts"] == "A" for row in rows))
                self.assertEqual(training.read_text(), "existing training rollout\n")


if __name__ == "__main__":
    unittest.main()
