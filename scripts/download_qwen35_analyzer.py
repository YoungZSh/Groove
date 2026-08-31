#!/usr/bin/env python3
"""Download the local Qwen3.5 35B-A3B FP8 Analyzer checkpoint reproducibly."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_REPO = "Qwen/Qwen3.5-35B-A3B-FP8"
DEFAULT_OUTPUT = Path("/root/siton-tmp/yzs/ckpts/Qwen3.5-35B-A3B-FP8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()

    if not any(os.environ.get(name) for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")):
        print("Warning: no proxy environment variable is set; using Hugging Face directly.")

    target = args.output_dir.resolve()
    target.mkdir(parents=True, exist_ok=True)
    location = snapshot_download(
        repo_id=args.repo,
        local_dir=str(target),
        max_workers=args.max_workers,
    )
    print(f"Analyzer checkpoint ready: {location}")


if __name__ == "__main__":
    main()
