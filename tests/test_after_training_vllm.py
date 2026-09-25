from contextlib import redirect_stdout
from importlib.util import module_from_spec, spec_from_file_location
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch


spec = spec_from_file_location("after_training_vllm", Path(__file__).parents[1] / "scripts/serve_2b_after_training.py")
controller = module_from_spec(spec)
spec.loader.exec_module(controller)


class AfterTrainingVllmTest(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        model = self.root / "model"
        model.mkdir()
        (model / "config.json").write_text("{}")
        (model / "model.safetensors").write_bytes(b"fixture")
        self.config = {
            "training_log": str(self.root / "training.log"),
            "checkpoint_root": str(self.root / "checkpoints"),
            "training_pid": 999999, "training_start_ticks": "original",
            "final_step": 123, "gpus": [0, 1, 2, 3], "ports": [8100, 8101, 8102, 8103],
            "python": sys.executable, "vllm_binary": sys.executable,
            "model_path": str(model), "served_model_name": "Qwen3.5-2B",
            "minimum_memory_mib": 72 * 1024, "kv_cache_gib": 69,
            "project_root": str(self.root), "output_dir": str(self.root / "service"),
        }

    def completion_files(self):
        Path(self.config["training_log"]).write_text("step:123 - metrics:1\nFinal validation metrics: {}\n")
        checkpoint = Path(self.config["checkpoint_root"]) / "global_step_123"
        (checkpoint / "actor").mkdir(parents=True)
        (checkpoint / "data.pt").write_bytes(b"state")
        (checkpoint.parent / "latest_checkpointed_iteration.txt").write_text("123")
        for rank in range(4):
            for prefix in ("model", "optim", "extra_state"):
                (checkpoint / "actor" / f"{prefix}_world_size_4_rank_{rank}.pt").write_bytes(b"state")

    def test_process_exit_without_completed_training_never_launches(self):
        Path(self.config["training_log"]).write_text("step:42 - metric:1\nTraceback: failed\n")
        with patch.object(controller, "process_identity", return_value=None):
            self.assertFalse(controller.training_status(self.config)["ready_to_launch"])

    def test_complete_checkpoint_and_process_exit_are_both_required(self):
        self.completion_files()
        with patch.object(controller, "process_identity", return_value="original"):
            self.assertFalse(controller.training_status(self.config)["ready_to_launch"])
        with patch.object(controller, "process_identity", return_value=None):
            self.assertTrue(controller.training_status(self.config)["ready_to_launch"])
            (Path(self.config["checkpoint_root"]) / "global_step_123/actor/optim_world_size_4_rank_3.pt").unlink()
            self.assertFalse(controller.training_status(self.config)["ready_to_launch"])

    def test_failed_training_requires_explicit_exit_override(self):
        Path(self.config["training_log"]).write_text("step:51 - metric:1\nRemoteDisconnected\n")
        with patch.object(controller, "process_identity", return_value=None):
            self.assertFalse(controller.training_status(self.config)["ready_to_launch"])
            self.config["allow_incomplete_training_exit"] = True
            self.assertTrue(controller.training_status(self.config)["ready_to_launch"])
        with patch.object(controller, "process_identity", return_value="original"):
            self.assertFalse(controller.training_status(self.config)["ready_to_launch"])

    def test_authorized_any_exit_restores_even_before_first_step(self):
        self.config["restore_on_any_training_exit"] = True
        for text in (None, "Traceback: failed during model initialization\n"):
            with self.subTest(log=text):
                if text is not None:
                    Path(self.config["training_log"]).write_text(text)
                with patch.object(controller, "process_identity", return_value=None):
                    state = controller.training_status(self.config)
                    self.assertTrue(state["ready_to_launch"])
                    self.assertFalse(state["completion_logged"])
                with patch.object(controller, "process_identity", return_value="original"):
                    self.assertFalse(controller.training_status(self.config)["ready_to_launch"])

    def test_any_exit_override_requires_boolean_true(self):
        self.config["restore_on_any_training_exit"] = "false"
        with patch.object(controller, "process_identity", return_value=None):
            self.assertFalse(controller.training_status(self.config)["ready_to_launch"])

    def test_reused_pid_is_not_the_original_training_process(self):
        self.completion_files()
        with patch.object(controller, "process_identity", return_value="different"):
            self.assertTrue(controller.training_status(self.config)["ready_to_launch"])

    def test_each_service_uses_tp_one_and_a_real_kv_cache_reservation(self):
        controller.validate_config(self.config)
        for slot in range(4):
            command = controller.service_command(self.config, slot, 1, 69)
            self.assertEqual(command[command.index("--tensor-parallel-size") + 1], "1")
            self.assertEqual(command[command.index("--port") + 1], str(8100 + slot))
            self.assertEqual(int(command[command.index("--kv-cache-memory-bytes") + 1]), 69 * 1024**3)
            self.assertEqual(command[3], self.config["model_path"])

    def test_retry_reduces_active_batch_without_reducing_cache(self):
        command = controller.service_command(self.config, 0, 2, 69)
        self.assertEqual(command[command.index("--max-num-seqs") + 1], "16")
        self.assertEqual(command[command.index("--max-num-batched-tokens") + 1], "8192")
        self.assertEqual(int(command[command.index("--kv-cache-memory-bytes") + 1]), 69 * 1024**3)

    def test_subset_keeps_original_checkpoint_world_size(self):
        self.completion_files()
        self.config.update(gpus=[1, 2, 3], ports=[8101, 8102, 8103], training_world_size=4)
        controller.validate_config(self.config)
        with patch.object(controller, "process_identity", return_value=None):
            self.assertTrue(controller.training_status(self.config)["ready_to_launch"])
            (Path(self.config["checkpoint_root"]) / "global_step_123/actor/model_world_size_4_rank_0.pt").unlink()
            self.assertFalse(controller.training_status(self.config)["ready_to_launch"])

    def test_rejects_adopting_gpu_outside_configured_subset(self):
        self.config.update(gpus=[1, 2, 3], ports=[8101, 8102, 8103], existing_services={"0": {}})
        with self.assertRaises(ValueError):
            controller.validate_config(self.config)
        with self.assertRaises(ValueError):
            controller.adopt_services(self.config)

    def test_adoption_checks_identity_command_and_device_selectors(self):
        command = [sys.executable, "-c", "import time; time.sleep(60)"]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="1", CUDA_DEVICE_ORDER="PCI_BUS_ID")
        child = subprocess.Popen(command, env=env, start_new_session=True)
        self.addCleanup(child.wait)
        self.addCleanup(child.terminate)
        identity = controller.process_identity(child.pid)
        record = {"pid": child.pid, "start_ticks": identity, "log": "existing.log"}
        self.config.update(gpus=[1], ports=[8101], existing_services={"1": record})
        with patch.object(controller, "service_command", return_value=command):
            adopted = controller.adopt_services(self.config)
            self.assertIsNone(adopted[1]["process"].poll())
            record["start_ticks"] = "reused-pid"
            with self.assertRaises(ValueError):
                controller.adopt_services(self.config)
            record["start_ticks"] = identity
            self.config.update(gpus=[2], existing_services={"2": record})
            with self.assertRaises(ValueError):
                controller.adopt_services(self.config)
        self.config.update(gpus=[1], existing_services={"1": record})
        with self.assertRaises(ValueError):
            controller.adopt_services(self.config)
        child.terminate()
        child.wait(timeout=5)
        self.assertIsNotNone(adopted[1]["process"].poll())

    def test_reused_adopted_pid_is_never_signalled(self):
        process = controller.ExistingProcess(900001, "original")
        with patch.object(controller, "process_identity", return_value="replacement"), \
                patch.object(controller.os, "killpg") as kill:
            self.assertIsNotNone(process.poll())
            controller.stop_service({"process": process, "start_ticks": "original"})
            kill.assert_not_called()

    def test_adopted_subset_is_ready_and_restarts_only_failed_configured_gpu(self):
        for failed_gpu in (None, 2):
            with self.subTest(failed_gpu=failed_gpu):
                self.config.update(gpus=[1, 2, 3], ports=[8101, 8102, 8103],
                                   output_dir=str(self.root / f"adopt-{failed_gpu}"))
                children = {gpu: {"process": SimpleNamespace(pid=900000 + gpu,
                                              poll=lambda gpu=gpu: 0 if gpu == failed_gpu else None),
                                  "start_ticks": "owned", "started": 0, "log": "existing.log",
                                  "probe": None, "ready": False} for gpu in (1, 2, 3)}
                cards = {gpu: {"pids": [] if gpu == failed_gpu else [900000 + gpu],
                               "used_mib": 73 * 1024} for gpu in (1, 2, 3)}
                output = io.StringIO()
                with patch.object(controller, "adopt_services", return_value=children), \
                        patch.object(controller, "training_status") as training, \
                        patch.object(controller, "gpu_state", return_value=cards) as gpu_state, \
                        patch.object(controller, "port_available", return_value=True), \
                        patch.object(controller, "probe_service", return_value="OK"), \
                        patch.object(controller, "process_identity", return_value="owned"), \
                        patch.object(controller, "stop_service"), \
                        patch.object(controller.subprocess, "Popen", return_value=SimpleNamespace(pid=910002)) as launch, \
                        patch.object(controller.time, "sleep", side_effect=KeyboardInterrupt), redirect_stdout(output):
                    with self.assertRaises(KeyboardInterrupt):
                        controller.supervise(self.config)
                training.assert_not_called()
                gpu_state.assert_called_once_with([1, 2, 3])
                self.assertEqual('"stage": "READY"' in output.getvalue(), failed_gpu is None)
                self.assertEqual(launch.call_count, int(failed_gpu is not None))
                if failed_gpu is not None:
                    self.assertEqual(launch.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "2")
                    command = launch.call_args.args[0]
                    self.assertEqual(command[command.index("--port") + 1], "8102")

    def test_supervisor_does_not_start_anything_while_training_is_running(self):
        with patch.object(controller, "training_status", return_value={"ready_to_launch": False, "training_process_alive": True}), \
                patch.object(controller.subprocess, "Popen") as launch, \
                patch.object(controller.time, "sleep", side_effect=KeyboardInterrupt), redirect_stdout(io.StringIO()):
            with self.assertRaises(KeyboardInterrupt):
                controller.supervise(self.config)
            launch.assert_not_called()

    def test_ready_requires_all_four_memory_floors_and_inference_probes(self):
        for under_floor in (True, False):
            with self.subTest(under_floor=under_floor):
                self.config["output_dir"] = str(self.root / f"service-{under_floor}")
                free = {gpu: {"uuid": f"gpu-{gpu}", "pids": [], "used_mib": 4, "total_mib": 81920} for gpu in range(4)}
                occupied = {gpu: {**free[gpu], "used_mib": 73 * 1024} for gpu in range(4)}
                if under_floor:
                    occupied[3]["used_mib"] = 72 * 1024 - 1
                processes = [SimpleNamespace(pid=900000 + gpu, poll=lambda: None) for gpu in range(4)]
                output = io.StringIO()
                with patch.object(controller, "training_status", return_value={"ready_to_launch": True}), \
                        patch.object(controller, "gpu_state", side_effect=[free, free, occupied]), \
                        patch.object(controller, "port_available", return_value=True), \
                        patch.object(controller, "process_identity", return_value="owned"), \
                        patch.object(controller, "probe_service", return_value="OK"), \
                        patch.object(controller, "stop_service"), \
                        patch.object(controller.subprocess, "Popen", side_effect=processes) as launch, \
                        patch.object(controller.time, "sleep", side_effect=[None, KeyboardInterrupt]), redirect_stdout(output):
                    with self.assertRaises(KeyboardInterrupt):
                        controller.supervise(self.config)
                self.assertEqual(launch.call_count, 4)
                for gpu, call in enumerate(launch.call_args_list):
                    self.assertEqual(call.kwargs["env"]["CUDA_VISIBLE_DEVICES"], str(gpu))
                    self.assertEqual(call.kwargs["env"]["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
                self.assertEqual('"stage": "READY"' in output.getvalue(), not under_floor)


if __name__ == "__main__":
    unittest.main()
