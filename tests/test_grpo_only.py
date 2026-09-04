from __future__ import annotations

import unittest

from groove.verl_trainer import GrooveRayPPOTrainer


class GRPOOnlyTrainerTest(unittest.TestCase):
    def test_disabled_opsd_does_not_construct_or_score_a_teacher(self):
        trainer = GrooveRayPPOTrainer.__new__(GrooveRayPPOTrainer)
        trainer.config = {"groove": {"enabled": False}}
        trainer._build_online_teacher_columns = lambda *_args: (_ for _ in ()).throw(
            AssertionError("GRPO-only mode must not build online evidence")
        )
        trainer._build_groove_teacher_batch = lambda *_args: (_ for _ in ()).throw(
            AssertionError("GRPO-only mode must not construct a Teacher batch")
        )
        trainer._compute_old_log_prob = lambda *_args: (_ for _ in ()).throw(
            AssertionError("GRPO-only mode must not run a Teacher forward")
        )

        batch = object()
        result, metrics = trainer._postprocess_advantages(batch, None, None)

        self.assertIs(result, batch)
        self.assertEqual(metrics, {})


if __name__ == "__main__":
    unittest.main()
