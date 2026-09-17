import os
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf

from groove.verl_entrypoint import configure_ray_memory_guard


GIB = 1024**3


class RayMemoryGuardTest(unittest.TestCase):
    def configure(self, total_gib, env):
        config = OmegaConf.create({"ray_kwargs": {"ray_init": {}}})
        with patch.dict(os.environ, env, clear=True), patch(
            "ray._private.utils.get_system_memory", return_value=total_gib * GIB
        ):
            result = configure_ray_memory_guard(config)
            effective_env = {
                key: os.environ[key]
                for key in ("RAY_memory_usage_threshold", "RAY_memory_monitor_refresh_ms")
            }
        return config, result, effective_env

    def test_uncapped_four_gpu_node_uses_actual_node_memory(self):
        config, result, env = self.configure(2048, {"RAY_NODE_MEMORY_CAP_GIB": "null"})
        self.assertIsNone(result["requested_cap_bytes"])
        self.assertEqual(result["headroom_bytes"], 0)
        self.assertEqual(result["threshold"], 0.95)
        self.assertEqual(result["trigger_bytes"], int(2048 * GIB * 0.95))
        self.assertEqual(env["RAY_memory_usage_threshold"], "0.950000000")
        self.assertEqual(config.ray_kwargs.ray_init.object_store_memory, 8 * GIB)
        self.assertFalse(config.ray_kwargs.ray_init.include_dashboard)

    def test_uncapped_mode_respects_the_memory_ray_detects_in_a_container(self):
        _, result, _ = self.configure(64, {"RAY_NODE_MEMORY_CAP_GIB": "none"})
        self.assertEqual(result["trigger_bytes"], int(64 * GIB * 0.95))

    def test_existing_fixed_cap_remains_available(self):
        for env in ({}, {"RAY_NODE_MEMORY_CAP_GIB": "220"}):
            with self.subTest(env=env):
                _, result, _ = self.configure(2048, env)
                self.assertEqual(result["requested_cap_bytes"], 220 * GIB)
                self.assertEqual(result["trigger_bytes"], 216 * GIB)

    def test_invalid_numeric_cap_is_not_silently_disabled(self):
        for cap in ("0", "-1", "invalid"):
            with self.subTest(cap=cap), self.assertRaises(ValueError):
                self.configure(2048, {"RAY_NODE_MEMORY_CAP_GIB": cap})

    def test_object_store_must_fit_within_available_memory(self):
        with self.assertRaises(ValueError):
            self.configure(4, {"RAY_NODE_MEMORY_CAP_GIB": "null"})


if __name__ == "__main__":
    unittest.main()
