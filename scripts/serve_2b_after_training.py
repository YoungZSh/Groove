#!/usr/bin/env python3
"""Wait for a completed training run, then supervise one TP=1 vLLM per GPU."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import io
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request


GIB = 1024**3


def process_identity(pid: int) -> str | None:
    """Read the start tick as well as PID, so a reused PID is never trusted."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except (OSError, IndexError):
        return None


class ExistingProcess:
    """Track an explicitly adopted service without requiring it to be our child."""

    def __init__(self, pid: int, start_ticks: str):
        self.pid = pid
        self.start_ticks = start_ticks

    def poll(self):
        return None if process_identity(self.pid) == self.start_ticks else 0

    def wait(self, timeout: float):
        deadline = time.monotonic() + timeout
        while self.poll() is None:
            if time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(str(self.pid), timeout)
            time.sleep(0.1)
        return 0


def adopt_services(config: dict) -> dict[int, dict]:
    """Validate every recorded identity before taking ownership of any service."""
    children = {}
    for gpu_key, record in config.get("existing_services", {}).items():
        gpu = int(gpu_key)
        if gpu not in config["gpus"]:
            raise ValueError(f"Cannot adopt unconfigured GPU {gpu}")
        slot = config["gpus"].index(gpu)
        pid, start_ticks = record["pid"], record["start_ticks"]
        proc = Path(f"/proc/{pid}")
        if not start_ticks or process_identity(pid) != start_ticks:
            raise ValueError(f"GPU {gpu} service identity changed: {pid}")
        if proc.stat().st_uid != os.getuid() or os.getpgid(pid) != pid:
            raise ValueError(f"GPU {gpu} service must be an owned process-group leader")
        command = [os.fsdecode(arg) for arg in proc.joinpath("cmdline").read_bytes().split(b"\0") if arg]
        expected = [service_command(config, slot, attempt, config["kv_cache_gib"]) for attempt in (1, 2)]
        if command not in expected:
            raise ValueError(f"GPU {gpu} service command does not match its configuration")
        # Inspect only device selectors; never emit the process environment.
        selectors = {entry.split(b"=", 1)[0]: entry.split(b"=", 1)[1]
                     for entry in proc.joinpath("environ").read_bytes().split(b"\0")
                     if entry.startswith((b"CUDA_VISIBLE_DEVICES=", b"CUDA_DEVICE_ORDER="))}
        if selectors.get(b"CUDA_VISIBLE_DEVICES") != str(gpu).encode() or selectors.get(b"CUDA_DEVICE_ORDER") != b"PCI_BUS_ID":
            raise ValueError(f"GPU {gpu} service device selectors do not match")
        if process_identity(pid) != start_ticks:
            raise ValueError(f"GPU {gpu} service exited during validation")
        children[gpu] = {"process": ExistingProcess(pid, start_ticks), "start_ticks": start_ticks,
                         "started": time.monotonic(), "log": record["log"], "probe": None, "ready": False}
    return children


def training_status(config: dict) -> dict:
    log = Path(config["training_log"])
    text = log.read_text(errors="replace") if log.exists() else ""
    steps = [int(value) for value in re.findall(r"\bstep:(\d+) -", text)]
    final_step = config["final_step"]
    checkpoint = Path(config["checkpoint_root"]) / f"global_step_{final_step}"
    expected = [checkpoint / "data.pt"]
    world_size = config.get("training_world_size", len(config["gpus"]))
    for rank in range(world_size):
        for prefix in ("model", "optim", "extra_state"):
            expected.append(checkpoint / "actor" / f"{prefix}_world_size_{world_size}_rank_{rank}.pt")
    checkpoint_ready = all(path.is_file() and path.stat().st_size > 0 for path in expected)
    tracker = Path(config["checkpoint_root"]) / "latest_checkpointed_iteration.txt"
    try:
        checkpoint_ready = checkpoint_ready and int(tracker.read_text().strip()) == final_step
    except (OSError, ValueError):
        checkpoint_ready = False
    complete = final_step in steps and "Final validation metrics:" in text and checkpoint_ready
    alive = process_identity(config["training_pid"]) == config["training_start_ticks"]
    authorized_incomplete_exit = bool(config.get("allow_incomplete_training_exit", False) and steps)
    return {
        "last_step": max(steps, default=None),
        "final_step": final_step,
        "checkpoint_ready": checkpoint_ready,
        "completion_logged": complete,
        "training_process_alive": alive,
        "authorized_incomplete_exit": authorized_incomplete_exit,
        "ready_to_launch": (complete or authorized_incomplete_exit) and not alive,
    }


def validate_config(config: dict) -> None:
    if not config["gpus"] or len(set(config["gpus"])) != len(config["gpus"]):
        raise ValueError("Configure one or more distinct GPUs")
    if any(type(gpu) is not int or gpu < 0 for gpu in config["gpus"]):
        raise ValueError("GPU indices must be nonnegative integers")
    if len(config["ports"]) != len(config["gpus"]) or len(set(config["ports"])) != len(config["ports"]):
        raise ValueError("Configure one distinct service port per GPU")
    if config.get("training_world_size", len(config["gpus"])) < 1:
        raise ValueError("Training world size must be positive")
    if any(int(gpu) not in config["gpus"] for gpu in config.get("existing_services", {})):
        raise ValueError("Existing services must belong to configured GPUs")
    if config["minimum_memory_mib"] < 72 * 1024:
        raise ValueError("The requested GPU memory floor is 72 GiB")
    if not 68 <= config["kv_cache_gib"] <= 70:
        raise ValueError("This 80 GiB GPU profile expects a 68–70 GiB KV cache")
    if any(not 1024 <= port <= 65535 for port in config["ports"]):
        raise ValueError("Service ports must be between 1024 and 65535")
    if config["training_pid"] <= 1 or not config["training_start_ticks"]:
        raise ValueError("A specific training process identity is required")
    for field in ("python", "vllm_binary"):
        if not Path(config[field]).is_file():
            raise FileNotFoundError(config[field])
    model = Path(config["model_path"])
    if not (model / "config.json").is_file() or not list(model.glob("*.safetensors")):
        raise ValueError("The local model config and weights must already exist")


def service_command(config: dict, slot: int, attempt: int, cache_gib: int) -> list[str]:
    # Retry with a smaller active batch if a process fails to initialize. The
    # actual KV cache reservation and memory-floor check remain in force.
    sequences, tokens = (32, 16384) if attempt == 1 else (16, 8192)
    return [
        config["python"], config["vllm_binary"], "serve", config["model_path"],
        "--served-model-name", config["served_model_name"],
        "--host", "127.0.0.1", "--port", str(config["ports"][slot]),
        "--tensor-parallel-size", "1", "--dtype", "bfloat16",
        "--max-model-len", "10240", "--max-num-seqs", str(sequences),
        "--max-num-batched-tokens", str(tokens), "--enable-chunked-prefill",
        "--enforce-eager", "--generation-config", "vllm",
        "--kv-cache-memory-bytes", str(cache_gib * GIB),
    ]


def gpu_state(gpus: list[int]) -> dict[int, dict]:
    result = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,memory.total,memory.used",
        "--format=csv,noheader,nounits", "-i", ",".join(map(str, gpus)),
    ], text=True, timeout=30)
    state = {
        int(row[0]): {"uuid": row[1].strip(), "total_mib": float(row[2]), "used_mib": float(row[3]), "pids": []}
        for row in csv.reader(io.StringIO(result))
    }
    processes = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits",
    ], text=True, timeout=30)
    by_uuid = {value["uuid"]: value for value in state.values()}
    for row in csv.reader(io.StringIO(processes)):
        if len(row) == 2 and row[0].strip() in by_uuid:
            by_uuid[row[0].strip()]["pids"].append(int(row[1]))
    return state


def port_available(port: int) -> bool:
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def probe_service(config: dict, slot: int, smoke: bool = False) -> str | None:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    base = f"http://127.0.0.1:{config['ports'][slot]}/v1"
    try:
        with opener.open(base + "/models", timeout=3) as response:
            models = json.load(response)
        if config["served_model_name"] not in {row["id"] for row in models.get("data", [])}:
            return None
        if not smoke:
            return "healthy"
        body = {
            "model": config["served_model_name"],
            "messages": [{"role": "user", "content": "Reply with OK."}],
            "temperature": 0, "max_completion_tokens": 16,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        request = urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"}, method="POST")
        with opener.open(request, timeout=30) as response:
            content = json.load(response)["choices"][0]["message"].get("content", "")
        return content.strip() or None
    except (OSError, ValueError, KeyError, IndexError, urllib.error.URLError):
        return None


def stop_service(service: dict) -> None:
    """Only terminate a process group created or explicitly adopted here."""
    process = service["process"]
    identity = process_identity(process.pid)
    if identity is not None and identity != service["start_ticks"]:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    # The CLI can exit before its engine children; finish that same group.
    identity = process_identity(process.pid)
    if identity is not None and identity != service["start_ticks"]:
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def supervise(config: dict) -> None:
    folder = Path(config["output_dir"])
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / "supervisor.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        children = adopt_services(config)
    except BaseException:
        lock.close()
        raise
    attempts = {gpu: int(gpu in children) for gpu in config["gpus"]}
    caches = {gpu: config["kv_cache_gib"] for gpu in config["gpus"]}
    activated = bool(children)
    last_stage = None

    def status(stage: str, **details):
        nonlocal last_stage
        record = {"stage": stage, "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                  "supervisor_pid": os.getpid(), **details}
        temporary = folder / "status.tmp"
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(folder / "status.json")
        if stage != last_stage:
            print(json.dumps(record), flush=True)
            last_stage = stage

    try:
        while True:
            try:
                if not activated:
                    progress = training_status(config)
                    if not progress["ready_to_launch"]:
                        stage = "WAITING_FOR_TRAINING" if progress["training_process_alive"] else "TRAINING_INCOMPLETE"
                        status(stage, training=progress)
                        time.sleep(config.get("poll_seconds", 2))
                        continue
                    cards = gpu_state(config["gpus"])
                    if any(card["pids"] or card["used_mib"] > 512 for card in cards.values()):
                        status("WAITING_FOR_GPU_RELEASE", training=progress, gpus=cards)
                        time.sleep(2)
                        continue
                    if not all(port_available(port) for port in config["ports"]):
                        status("WAITING_FOR_FREE_PORTS", ports=config["ports"])
                        time.sleep(2)
                        continue
                    activated = True
                    print(f"Training exit condition satisfied; starting {len(config['gpus'])} TP=1 vLLM services", flush=True)

                cards = gpu_state(config["gpus"])
                for slot, gpu in enumerate(config["gpus"]):
                    service = children.get(gpu)
                    if service and service["process"].poll() is not None:
                        stop_service(service)
                        children.pop(gpu)
                        service = None
                    if service is None:
                        if cards[gpu]["pids"] or not port_available(config["ports"][slot]):
                            continue
                        attempts[gpu] += 1
                        command = service_command(config, slot, attempts[gpu], caches[gpu])
                        env = dict(os.environ)
                        # This vLLM release parses visible device IDs as integers.
                        env.update(CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID", OMP_NUM_THREADS="4",
                                   PYTHONUNBUFFERED="1", VLLM_WORKER_MULTIPROC_METHOD="spawn")
                        env.pop("RAY_ADDRESS", None)
                        env.pop("VLLM_USE_V1", None)
                        library = str(Path(config["python"]).parent.parent / "lib")
                        env["LD_LIBRARY_PATH"] = library + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
                        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                        log = folder / f"gpu{gpu}-attempt{attempts[gpu]}-{stamp}.log"
                        with log.open("xb") as handle:
                            process = subprocess.Popen(command, cwd=config["project_root"], env=env,
                                                       stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                        children[gpu] = {"process": process, "start_ticks": process_identity(process.pid),
                                         "started": time.monotonic(), "log": str(log), "probe": None, "ready": False}
                        print(json.dumps({"gpu": gpu, "pid": process.pid, "command": command, "log": str(log)}), flush=True)
                        continue
                    healthy = probe_service(config, slot)
                    if healthy and service.get("healthy_since") is None:
                        service["healthy_since"] = time.monotonic()
                    if healthy and service["probe"] is None:
                        service["probe"] = probe_service(config, slot, smoke=True)
                    service["ready"] = bool(healthy and service["probe"] and cards[gpu]["used_mib"] >= config["minimum_memory_mib"])
                    if (healthy and service["probe"] and not service["ready"] and caches[gpu] < 70
                            and time.monotonic() - service["healthy_since"] > 20):
                        caches[gpu] += 1
                        stop_service(service)
                        children.pop(gpu)
                    elif not healthy and time.monotonic() - service["started"] > config.get("startup_timeout_seconds", 900):
                        stop_service(service)
                        children.pop(gpu)

                ready = len(children) == len(config["gpus"]) and all(service["ready"] for service in children.values())
                details = {str(gpu): {"pid": service["process"].pid, "port": config["ports"][config["gpus"].index(gpu)],
                                      "memory_mib": cards[gpu]["used_mib"], "required_memory_mib": config["minimum_memory_mib"],
                                      "start_ticks": service["start_ticks"],
                                      "ready": service["ready"], "inference_probe": service["probe"], "log": service["log"]}
                           for gpu, service in children.items()}
                status("READY" if ready else "STARTING_OR_RECOVERING", services=details)
                time.sleep(10 if ready else 2)
            except Exception as error:
                status("RETRYING_AFTER_ERROR", error=f"{type(error).__name__}: {error}")
                time.sleep(5)
    finally:
        for service in children.values():
            stop_service(service)
        status("STOPPED")
        lock.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--check", action="store_true", help="Validate configuration without starting any process")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    validate_config(config)
    if args.check:
        print(json.dumps({"training": training_status(config),
                          "commands": [service_command(config, slot, 1, config["kv_cache_gib"]) for slot in range(len(config["gpus"]))]}, indent=2))
        return
    def interrupted(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    try:
        supervise(config)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
