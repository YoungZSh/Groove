"""Optional worker-owned stack dumps for diagnosing stalled training phases."""

import faulthandler
import os
from pathlib import Path
import signal

_stack_log = None


def setup_worker_diagnostics():
    directory = os.environ.get("GROOVE_WORKER_DEBUG_DIR")
    if not directory:
        return
    global _stack_log
    Path(directory).mkdir(parents=True, exist_ok=True)
    _stack_log = (Path(directory) / f"{os.getpid()}.log").open("a")
    faulthandler.register(signal.SIGUSR2, file=_stack_log, all_threads=True)
