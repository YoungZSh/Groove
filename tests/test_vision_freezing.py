from copy import deepcopy
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from verl.workers.config.actor import FSDPActorConfig
from verl.workers.config.engine import FSDPEngineConfig
from verl.workers.utils.vision_freezing import (
    capture_vision_update,
    freeze_qwen35_vision,
    parameter_group,
    verify_vision_update,
)


class TinyVisual(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Linear(6, 8)
        self.pos_embed = nn.Parameter(torch.randn(1, 8))
        self.blocks = nn.Sequential(nn.Linear(8, 8), nn.Tanh())
        self.merger = nn.Linear(8, 8)

    def forward(self, pixels):
        return self.merger(self.blocks(self.patch_embed(pixels) + self.pos_embed))


class TinyQwen35(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="qwen3_5")
        self.model = nn.Module()
        self.model.visual = TinyVisual()
        self.model.language_model = nn.Linear(8, 5)
        self.lm_head = nn.Linear(5, 3)

    def forward(self, pixels):
        return self.lm_head(self.model.language_model(self.model.visual(pixels)))


class VisionFreezingTest(unittest.TestCase):
    def test_backbone_remains_identical_while_merger_and_language_learn(self):
        torch.manual_seed(12)
        model = TinyQwen35()
        report = freeze_qwen35_vision(model, train_merger=True)
        self.assertEqual(report["parameters"]["vit"]["trainable"], 0)
        original = deepcopy(model.state_dict())
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=.01, weight_decay=.1)
        for _ in range(3):
            optimizer.zero_grad()
            model(torch.randn(4, 6)).square().mean().backward()
            before = capture_vision_update(model)
            optimizer.step()
            audit = verify_vision_update(model, before)
            self.assertTrue(audit["frozen_weights_unchanged"])
            for group in ("merger", "llm"):
                self.assertGreater(audit["gradient_norms"][group], 0)
                self.assertGreater(audit["changed_parameter_samples"][group], 0)
        for name, parameter in model.named_parameters():
            if parameter_group(name) == "vit":
                self.assertIsNone(parameter.grad)
                self.assertTrue(torch.equal(parameter, original[name]))

    def test_freeze_entire_visual_without_merger_override(self):
        model = TinyQwen35()
        freeze_qwen35_vision(model, train_merger=False)
        self.assertTrue(all(not p.requires_grad for p in model.model.visual.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.model.language_model.parameters()))

    def test_invalid_architecture_or_missing_merger_fails_before_freezing(self):
        for change in ("type", "merger"):
            model = TinyQwen35()
            if change == "type":
                model.config.model_type = "other"
            else:
                del model.model.visual.merger
            with self.assertRaises(ValueError):
                freeze_qwen35_vision(model, train_merger=True)
            self.assertTrue(all(p.requires_grad for p in model.parameters()))

    def test_audit_catches_a_frozen_gradient_or_weight_change(self):
        model = TinyQwen35()
        freeze_qwen35_vision(model, train_merger=True)
        parameter = model.model.visual.patch_embed.weight
        parameter.grad = torch.ones_like(parameter)
        with self.assertRaisesRegex(RuntimeError, "gradient"):
            capture_vision_update(model)
        parameter.grad = None
        before = capture_vision_update(model)
        with torch.no_grad():
            parameter[0, 0].add_(1)
        with self.assertRaisesRegex(RuntimeError, "changed"):
            verify_vision_update(model, before)

    def test_fsdp_names_and_configuration_reach_engine(self):
        self.assertEqual(parameter_group("_fsdp_wrapped_module.model.visual.blocks.0._fsdp_wrapped_module.weight"), "vit")
        self.assertEqual(parameter_group("_fsdp_wrapped_module.model.visual.merger.weight"), "merger")
        cfg = FSDPActorConfig(rollout_n=8, use_dynamic_bsz=True, freeze_vision_tower=True,
                              train_vision_merger=True, fsdp_config=FSDPEngineConfig(use_orig_params=True))
        self.assertTrue(cfg.engine.freeze_vision_tower)
        self.assertTrue(cfg.engine.train_vision_merger)
        with self.assertRaisesRegex(ValueError, "use_orig_params"):
            FSDPActorConfig(rollout_n=8, use_dynamic_bsz=True, freeze_vision_tower=True)
        with self.assertRaisesRegex(ValueError, "requires freeze"):
            FSDPActorConfig(rollout_n=8, use_dynamic_bsz=True, train_vision_merger=True)

    def test_real_qwen35_visual_forward_preserves_merger_gradient(self):
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

        config = Qwen3_5VisionConfig(depth=2, hidden_size=32, intermediate_size=64, num_heads=4,
                                    patch_size=2, spatial_merge_size=2, temporal_patch_size=1,
                                    out_hidden_size=8, num_position_embeddings=64)
        config._attn_implementation = "eager"
        model = TinyQwen35()
        model.model.visual = Qwen3_5VisionModel(config)
        model.model.visual.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        freeze_qwen35_vision(model, train_merger=True)
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=.01)
        visual = model.model.visual(torch.randn(16, 12), grid_thw=torch.tensor([[1, 4, 4]]))
        merged = visual.pooler_output
        model.lm_head(model.model.language_model(merged)).square().mean().backward()
        before = capture_vision_update(model)
        optimizer.step()
        report = verify_vision_update(model, before)
        self.assertEqual(report["gradient_norms"]["vit"], 0)
        self.assertGreater(report["gradient_norms"]["merger"], 0)
        self.assertGreater(report["gradient_norms"]["llm"], 0)
        self.assertTrue(report["frozen_weights_unchanged"])


if __name__ == "__main__":
    unittest.main()
