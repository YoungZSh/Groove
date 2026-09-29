#!/usr/bin/env python3
"""Run an existing training launcher with a separate tmux recovery watchdog.

The reviewed JSON configuration records exact service identities and a copy of
their original launcher. No global Ray stop or unscoped process killing is used.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time

from serve_2b_after_training import gpu_state, port_available, probe_service, process_identity


MARKER = "GROOVE_HANDOFF_ID"


def save(folder, name, **record):
    path = Path(folder) / f"{name}.json"
    record["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(path)


def owned_processes():
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            if path.stat().st_uid != os.getuid():
                continue
            pid = int(path.name)
            tick = process_identity(pid)
            if tick:
                yield pid, tick, path
        except OSError:
            continue


def tagged_processes(marker):
    if not marker:
        raise ValueError("A nonempty unique training marker is required")
    expected = f"{MARKER}={marker}".encode()
    result = {}
    for pid, tick, path in owned_processes():
        try:
            if pid != os.getpid() and expected in (path / "environ").read_bytes().split(b"\0"):
                result[pid] = tick
        except OSError:
            continue
    return result


def signal_records(records, sig):
    for pid, tick in records.items():
        if tick is not None and process_identity(pid) == tick:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass


def cleanup_training(marker):
    records = tagged_processes(marker)
    signal_records(records, signal.SIGTERM)
    deadline = time.monotonic() + 15
    while tagged_processes(marker) and time.monotonic() < deadline:
        time.sleep(.5)
    # Re-scan to include owned Ray children created during shutdown.
    signal_records(tagged_processes(marker), signal.SIGKILL)


def pane_info(service):
    result = subprocess.run(
        ["tmux", "display-message", "-p", "-t", service["pane"], "#{pane_pid} #{pane_dead}"],
        capture_output=True, text=True,
    )
    if result.returncode or not result.stdout.strip():
        return None
    pid, dead = map(int, result.stdout.split())
    return pid, bool(dead)


def resume_checkpoint(config):
    """Accept only an explicit complete checkpoint, while keeping a new output run."""
    env = config["training_env"]
    mode = env.get("RESUME_MODE", "disable")
    if mode == "disable":
        return None
    if mode != "resume_path":
        raise ValueError("Service handoff resume requires RESUME_MODE=resume_path")
    source = Path(env.get("RESUME_FROM_PATH", ""))
    if (not source.is_absolute() or not source.name.startswith("global_step_")
            or not source.name.removeprefix("global_step_").isdigit()):
        raise ValueError("RESUME_FROM_PATH must name an absolute global_step_N checkpoint")
    world_size = len(config["gpus"])
    required = [source / "data.pt", source / "actor/fsdp_config.json"]
    required.extend(source / "actor" / f"{kind}_world_size_{world_size}_rank_{rank}.pt"
                    for rank in range(world_size) for kind in ("model", "optim", "extra_state"))
    if any(not path.is_file() or path.stat().st_size == 0 for path in required):
        raise ValueError("Resume requires complete model, optimizer, RNG and dataloader checkpoints")
    if json.loads((source / "actor/fsdp_config.json").read_text())["world_size"] != world_size:
        raise ValueError("Resume checkpoint world size differs from the selected GPUs")
    best = source.parent / "best_checkpoint"
    if (best / "metadata.json").exists():
        metadata = json.loads((best / "metadata.json").read_text())
        if metadata["path"] != f"global_step_{int(metadata['step'])}":
            raise ValueError("Invalid inherited best checkpoint path")
        snapshot = best / metadata["path"]
        if not (snapshot / "data.pt").is_file() or not (snapshot / "actor").is_dir():
            raise ValueError("Inherited best checkpoint is incomplete")
    return source


def inherit_best_checkpoint(config):
    source = resume_checkpoint(config)
    if source is None or not (source.parent / "best_checkpoint/metadata.json").is_file():
        return
    destination = Path(config["project_root"]) / "checkpoints" / config["experiment"]
    # Separate copies preserve the old run when the new best is replaced later.
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source.parent / "best_checkpoint", destination / "best_checkpoint")
    save(config["output_dir"], "resume_lineage", source_checkpoint=str(source),
         inherited_best=json.loads((destination / "best_checkpoint/metadata.json").read_text()))


def validate(config):
    if len(config["gpus"]) != 2 or len(set(config["gpus"])) != 2:
        raise ValueError("This handoff requires exactly two distinct GPUs")
    if sorted(s["gpu"] for s in config["services"]) != sorted(config["gpus"]):
        raise ValueError("Service GPUs must exactly match training GPUs")
    if not config["experiment"] or not config["marker"]:
        raise ValueError("Experiment and process marker are required")
    if config["training_env"].get("CUDA_VISIBLE_DEVICES") != ",".join(map(str, config["gpus"])):
        raise ValueError("Training GPU selectors must match the service handoff")
    if config["training_env"].get("EXPERIMENT_NAME") != config["experiment"]:
        raise ValueError("Training experiment must match the recovery configuration")
    if Path(config["ray_temp_dir"]).exists():
        raise ValueError("Use a new, isolated Ray temporary directory")
    snapshot = Path(config["service_script"])
    if hashlib.sha256(snapshot.read_bytes()).hexdigest() != config["service_script_sha256"]:
        raise ValueError("Original service launcher snapshot changed")
    for s in config["services"]:
        pid = s["pid"]
        if process_identity(pid) != s["start_ticks"]:
            raise ValueError(f"GPU {s['gpu']} original process identity changed")
        path = Path(f"/proc/{pid}")
        if path.stat().st_uid != os.getuid() or os.getpgid(pid) != pid:
            raise ValueError("Original service must be an owned process-group leader")
        command = [os.fsdecode(x) for x in (path / "cmdline").read_bytes().split(b"\0") if x]
        if command != s["command"] or pane_info(s) != (pid, False):
            raise ValueError("Original service command or tmux pane changed")
        selector = f"CUDA_VISIBLE_DEVICES={s['uuid']}".encode()
        if selector not in (path / "environ").read_bytes().split(b"\0"):
            raise ValueError("Original service GPU UUID changed")
    checkpoint = Path(config["project_root"]) / "checkpoints" / config["experiment"]
    if checkpoint.exists():
        raise ValueError("Use a new experiment name")
    resume_checkpoint(config)


def stop_original(service):
    pid = service["pid"]
    if process_identity(pid) != service["start_ticks"]:
        raise RuntimeError("Original service identity changed before stopping")
    records = {}
    for candidate, tick, _ in owned_processes():
        try:
            if os.getpgid(candidate) == pid:
                records[candidate] = tick
        except ProcessLookupError:
            pass
    # Keep the original pane so recovery can respawn exactly that service.
    subprocess.run(["tmux", "set-option", "-w", "-t", service["pane"], "remain-on-exit", "on"], check=True)
    signal_records(records, signal.SIGTERM)
    deadline = time.monotonic() + 20
    while any(process_identity(p) == t for p, t in records.items()) and time.monotonic() < deadline:
        time.sleep(.5)
    signal_records(records, signal.SIGKILL)


def cards_free(gpus):
    return all(not c["pids"] and c["used_mib"] < 512 for c in gpu_state(gpus).values())


def run_training(config, config_path):
    validate(config)
    folder = Path(config["output_dir"])
    lock = (folder / "runner.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Finish checkpoint preparation before arming the watchdog or stopping services.
    inherit_best_checkpoint(config)
    runner = {"pid": os.getpid(), "start_ticks": process_identity(os.getpid())}
    save(folder, "runner", **runner)
    watchdog_command = shlex.join([sys.executable, "-u", str(Path(__file__).resolve()),
                                  "--config", str(config_path), "--watch"])
    watchdog_command += " >> " + shlex.quote(str(folder / "recovery.log")) + " 2>&1"
    subprocess.run(["tmux", "new-session", "-d", "-s", config["watchdog_session"], watchdog_command], check=True)
    deadline = time.monotonic() + 20
    while not (folder / "armed.json").exists():
        if time.monotonic() > deadline:
            raise RuntimeError("Recovery watchdog did not arm; original services remain running")
        time.sleep(.2)
    armed = json.loads((folder / "armed.json").read_text())
    if process_identity(armed["pid"]) != armed["start_ticks"]:
        raise RuntimeError("Recovery watchdog exited before service handoff")
    interrupted = []
    def request_stop(signum, _frame):
        interrupted.append(signum)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, request_stop)
    result = 1
    try:
        for service in config["services"]:
            if interrupted:
                return 128 + interrupted[0]
            stop_original(service)
        deadline = time.monotonic() + 120
        while not cards_free(config["gpus"]):
            if interrupted:
                return 128 + interrupted[0]
            if time.monotonic() > deadline:
                raise RuntimeError("Selected GPUs did not become free")
            time.sleep(1)
        env = dict(os.environ)
        env.update(config["training_env"])
        env.update({MARKER: config["marker"], "RAY_ADDRESS": "local", "GROOVE_DRY_RUN": "false"})
        command = ["bash", str(Path(config["project_root"]) / "scripts/train_a800_2gpu_nokl.sh"),
                   "++ray_kwargs.ray_init.address=local",
                   "++ray_kwargs.ray_init.num_gpus=2", "++ray_kwargs.ray_init.num_cpus=32",
                   f"++ray_kwargs.ray_init._temp_dir={config['ray_temp_dir']}",
                   f"++ray_kwargs.ray_init.runtime_env.env_vars.{MARKER}={config['marker']}"]
        with (folder / "launcher.log").open("ab") as log:
            child = subprocess.Popen(command, cwd=config["project_root"], env=env,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            save(folder, "training", pid=child.pid, start_ticks=process_identity(child.pid),
                 command=command, experiment=config["experiment"])
            while True:
                if interrupted:
                    result = 128 + interrupted[0]
                    break
                try:
                    result = child.wait(timeout=2)
                    break
                except subprocess.TimeoutExpired:
                    pass
    finally:
        save(folder, "training_exit", returncode=result, signals=interrupted)
        # The independent watchdog also handles SIGKILL of this runner.
        lock.close()
    return result


def runner_exited(folder):
    runner = json.loads((Path(folder) / "runner.json").read_text())
    return process_identity(runner["pid"]) != runner["start_ticks"]


def restart_service(config, service):
    command = shlex.join(["bash", config["service_script"], str(service["gpu"])])
    info = pane_info(service)
    if info is not None:
        if not info[1]:
            return False
        subprocess.run(["tmux", "respawn-pane", "-t", service["pane"], command], check=True)
    else:
        subprocess.run(["tmux", "new-session", "-d", "-s", service["session"], command], check=True)
        service["pane"] = subprocess.check_output(
            ["tmux", "display-message", "-p", "-t", service["session"], "#{pane_id}"], text=True).strip()
    return True


def watch(config):
    folder = Path(config["output_dir"])
    with (folder / "recovery.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _watch(config)


def _watch(config):
    folder = Path(config["output_dir"])
    save(folder, "armed", pid=os.getpid(), start_ticks=process_identity(os.getpid()))
    save(folder, "recovery_status", stage="WAITING_FOR_TRAINING_EXIT")
    while not runner_exited(folder):
        time.sleep(2)
    save(folder, "recovery_status", stage="CLEANING_TRAINING_PROCESSES")
    cleanup_training(config["marker"])
    started = {}
    attempts = {}
    probe_config = {"ports": [s["port"] for s in config["services"]],
                    "served_model_name": config["served_model_name"]}
    while True:
        try:
            cards = gpu_state(config["gpus"])
            ready = []
            for slot, service in enumerate(config["services"]):
                gpu = service["gpu"]
                info = pane_info(service)
                healthy = info is not None and not info[1] and probe_service(probe_config, slot) is not None
                ready.append(bool(healthy))
                if healthy:
                    continue
                if gpu in started and time.monotonic() - started[gpu]["time"] > 900:
                    pid = next(iter(started[gpu]["records"]))
                    if process_identity(pid) == started[gpu]["records"][pid]:
                        stop_original({**service, "pid": pid,
                                       "start_ticks": started[gpu]["records"][pid]})
                    started.pop(gpu)
                if not cards[gpu]["pids"] and cards[gpu]["used_mib"] < 512 and port_available(service["port"]):
                    if restart_service(config, service):
                        pid, _ = pane_info(service)
                        records = {pid: process_identity(pid)}
                        started[gpu] = {"time": time.monotonic(), "records": records}
                        attempts[gpu] = attempts.get(gpu, 0) + 1
            if all(ready):
                probes = [probe_service(probe_config, slot, smoke=True) for slot in range(len(ready))]
                if all(probes):
                    save(folder, "recovery_status", stage="RESTORED", probes=probes,
                         ports=probe_config["ports"], attempts=attempts)
                    return
            save(folder, "recovery_status", stage="RESTORING_SERVICES", ready=ready, attempts=attempts)
        except Exception as exc:
            save(folder, "recovery_status", stage="RETRYING_RECOVERY", error=f"{type(exc).__name__}: {exc}")
        time.sleep(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.check:
        validate(config)
        print("Handoff configuration and original service identities verified")
    elif args.watch:
        watch(config)
    else:
        raise SystemExit(run_training(config, args.config.resolve()))


if __name__ == "__main__":
    main()
