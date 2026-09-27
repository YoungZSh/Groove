#!/usr/bin/env python3
"""Create a compact Markdown handoff from a visual-evidence training run."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def metric(record: str, key: str) -> float | None:
    match = re.search(
        rf"{re.escape(key)}:(?:np\.float64\()?([-+0-9.eE]+)", record
    )
    return float(match.group(1)) if match else None


def step_records(log_text: str) -> dict[int, str]:
    chunks: dict[int, list[str]] = {}
    cleaned = ANSI.sub("", log_text)
    for match in re.finditer(r"step:(\d+) - (.*?)(?=step:\d+ -|\Z)", cleaned, re.DOTALL):
        step = int(match.group(1))
        chunks.setdefault(step, []).append(match.group(2))

    # Ray's console logger can split one metrics dictionary across multiple
    # lines, each repeating `step:N -`; reconstruct the complete record first.
    return {step: " ".join(parts) for step, parts in chunks.items()}


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=3)
    args = parser.parse_args()

    records = step_records(args.log.read_text(encoding="utf-8", errors="replace"))
    if not records:
        raise RuntimeError(f"No completed step records found in {args.log}")
    steps = sorted(records)

    evidence_paths = sorted(args.evidence_dir.glob("*/evidence.json"))
    evidence = [json.loads(path.read_text(encoding="utf-8")) for path in evidence_paths]
    statuses = Counter(item["status"] for item in evidence)
    routes = Counter(
        ((item.get("focus") or {}).get("crucial_evidence_type", "unknown"))
        for item in evidence
    )

    def values(key: str) -> list[float]:
        return [value for record in records.values() if (value := metric(record, key)) is not None]

    tracker = args.checkpoint_dir / "latest_checkpointed_iteration.txt"
    checkpoint_step = tracker.read_text(encoding="utf-8").strip() if tracker.exists() else "unknown"
    checkpoint_files = sorted((args.checkpoint_dir / f"global_step_{checkpoint_step}" / "actor").glob("*.pt"))

    lines = [
        "# Visual-evidence training progress",
        "",
        f"- Completed logged steps: {steps[0]}–{steps[-1]}",
        f"- Latest resumable checkpoint: `global_step_{checkpoint_step}`",
        "- Configuration: Batch 8 × 8 rollouts = 64 trajectories/step; vLLM `max_num_seqs=64`; FP32 AdamW; no CPU offload; KL=0.001; no entropy.",
        "",
        "## Aggregate signal",
        "",
        f"- Evidence ready rate: {statuses['ready']}/{len(evidence)} ({statuses['ready'] / max(len(evidence), 1):.1%})",
        f"- Evidence types: visual {routes['visual']}, text {routes['text']}, unknown {routes['unknown']}",
        f"- Mean signed OPSD advantage: {mean(values('actor/groove_opsd_advantage_mean')):.6f}",
        f"- Mean OPSD active-token ratio: {mean(values('actor/groove_opsd_active_token_ratio')):.1%}",
        f"- Mean positive/negative OPSD token fractions: "
        f"{mean(values('actor/groove_opsd_positive_token_fraction')):.1%} / "
        f"{mean(values('actor/groove_opsd_negative_token_fraction')):.1%}",
        f"- Mean answer reward: {mean(values('reward/answer_reward_mean')):.3f}",
        "",
        "## Resource envelope",
        "",
        f"- Max PyTorch allocated memory: {max(values('perf/max_memory_allocated_gb')):.2f} GB/card",
        f"- Max PyTorch reserved memory: {max(values('perf/max_memory_reserved_gb')):.2f} GB/card",
        f"- Max reported CPU memory: {max(values('perf/cpu_memory_used_gb')):.2f} GB",
        f"- Mean step time: {mean(values('timing_s/step')):.1f} s",
        f"- Mean throughput: {mean(values('perf/throughput')):.1f} tokens/s",
        "",
        "## Checkpoint files",
        "",
    ]
    for file in checkpoint_files:
        lines.append(f"- `{file.name}` — {file.stat().st_size / 2**30:.2f} GiB")

    lines.extend(["", "## Evidence samples", ""])
    ready = [item for item in evidence if item["status"] == "ready"]
    for index, item in enumerate(ready[-args.sample_count :], 1):
        focus = item.get("focus") or {}
        crops = item.get("crops", [])
        lines.extend(
            [
                f"### Sample {index}: `{item['uid']}`",
                "",
                f"- Crucial evidence type/route: `{focus.get('crucial_evidence_type')}` / `{focus.get('tool_route')}`",
                f"- Teacher focus: {focus.get('visible_focus_instruction')}",
                f"- Crops: {len(crops)}",
            ]
        )
        for crop in crops:
            lines.append(f"  - [{Path(crop['path']).name}]({crop['path']})")
        lines.append("")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
