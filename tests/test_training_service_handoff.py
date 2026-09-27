from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = spec_from_file_location("handoff", SCRIPTS / "training_service_handoff.py")
handoff = module_from_spec(spec)
spec.loader.exec_module(handoff)
sys.path.pop(0)


class TrainingServiceHandoffTest(unittest.TestCase):
    def test_cleanup_only_signals_exact_tagged_process_and_preserves_other_process(self):
        marker = "handoff-test-" + str(os.getpid())
        tagged = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                                  env={**os.environ, handoff.MARKER: marker})
        unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            self.assertIn(tagged.pid, handoff.tagged_processes(marker))
            self.assertNotIn(unrelated.pid, handoff.tagged_processes(marker))
            handoff.cleanup_training(marker)
            tagged.wait(timeout=5)
            self.assertIsNone(unrelated.poll())
        finally:
            for child in (tagged, unrelated):
                if child.poll() is None:
                    child.kill()
                child.wait(timeout=5)

    def test_reused_pid_is_not_signalled(self):
        with patch.object(handoff, "process_identity", return_value="replacement"), \
                patch.object(handoff.os, "kill") as kill:
            handoff.signal_records({900001: "original"}, signal.SIGKILL)
            kill.assert_not_called()

    def test_empty_marker_is_rejected(self):
        with self.assertRaises(ValueError):
            handoff.tagged_processes("")

    def test_exit_or_killed_runner_triggers_recovery_without_completed_checkpoint(self):
        with TemporaryDirectory() as folder:
            handoff.save(folder, "runner", pid=900001, start_ticks="original")
            with patch.object(handoff, "process_identity", return_value="original"):
                self.assertFalse(handoff.runner_exited(folder))
            for identity in (None, "replacement"):
                with patch.object(handoff, "process_identity", return_value=identity):
                    self.assertTrue(handoff.runner_exited(folder))

    def test_existing_live_service_is_never_respawned(self):
        service = {"gpu": 3, "pane": "%1", "session": "service3"}
        with patch.object(handoff, "pane_info", return_value=(12, False)), \
                patch.object(handoff.subprocess, "run") as run:
            self.assertFalse(handoff.restart_service({"service_script": "/original serve.sh"}, service))
            run.assert_not_called()

    def test_dead_service_restores_original_script_gpu_and_pane(self):
        service = {"gpu": 3, "pane": "%1", "session": "service3"}
        with patch.object(handoff, "pane_info", return_value=(12, True)), \
                patch.object(handoff.subprocess, "run") as run:
            self.assertTrue(handoff.restart_service({"service_script": "/original serve.sh"}, service))
            self.assertEqual(run.call_args.args[0],
                             ["tmux", "respawn-pane", "-t", "%1", "bash '/original serve.sh' 3"])

    def test_watchdog_waits_then_cleans_and_checks_restored_inference(self):
        with TemporaryDirectory() as folder:
            config = {"output_dir": folder, "gpus": [0, 3], "marker": "unique-run",
                      "served_model_name": "Qwen3.5-2B", "services": [
                          {"gpu": 0, "port": 8000}, {"gpu": 3, "port": 8003}]}
            with patch.object(handoff, "runner_exited", side_effect=[False, True]), \
                    patch.object(handoff.time, "sleep"), \
                    patch.object(handoff, "cleanup_training") as cleanup, \
                    patch.object(handoff, "gpu_state", return_value={0: {}, 3: {}}), \
                    patch.object(handoff, "pane_info", return_value=(12, False)), \
                    patch.object(handoff, "probe_service", return_value="OK") as probe:
                handoff.watch(config)
            cleanup.assert_called_once_with("unique-run")
            self.assertEqual(probe.call_count, 4)
            result = json.loads((Path(folder) / "recovery_status.json").read_text())
            self.assertEqual(result["stage"], "RESTORED")
            self.assertEqual(result["ports"], [8000, 8003])


if __name__ == "__main__":
    unittest.main()
