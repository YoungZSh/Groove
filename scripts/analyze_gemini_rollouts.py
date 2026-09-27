#!/usr/bin/env python3
"""Analyze prepared GroupRollout JSONL with Gemini; never generate new rollouts."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import fields, replace
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from groove.evidence import EvidenceBuilderConfig, TeacherEvidenceBuilder
from groove.gemini_analyzer import GeminiAnalyzerConfig, GeminiAPIAnalyzer, NoDetectorFallback
from groove.schemas import GroupRollout


def load_groups(path: Path) -> list[GroupRollout]:
    groups = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            group = GroupRollout.model_validate_json(line)
        except ValueError:
            raise ValueError(f"Line {line_number} must be a GroupRollout record; see docs/GEMINI_ANALYZER.md") from None
        if not group.rollouts:
            raise ValueError(f"Line {line_number} has no rollouts")
        image_path = group.image_path
        if not image_path.is_absolute():
            image_path = path.parent / image_path
        with Image.open(image_path) as image:
            image.verify()
        groups.append(group.model_copy(update={"image_path": image_path.resolve()}))
    if not groups:
        raise ValueError("The groups file is empty")
    return groups


def _write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", type=Path, required=True, help="One GroupRollout JSON object per line")
    parser.add_argument("--output-dir", type=Path, required=True, help="A NEW directory; existing paths are refused")
    parser.add_argument("--env-file", type=Path, default=Path(__file__).resolve().parents[1] / ".env")
    parser.add_argument("--limit", type=int, help="Analyze at most this many prepared groups")
    parser.add_argument("--max-tool-rounds", type=int, default=3)
    parser.add_argument("--max-completion-tokens", type=int, default=16384)
    parser.add_argument("--dry-run", action="store_true", help="Validate input and configuration without an API call or output writes")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.output_dir.exists():
        parser.error("--output-dir already exists; choose a new directory to preserve previous evidence")
    config = replace(
        GeminiAnalyzerConfig.from_env(args.env_file),
        max_tool_rounds=args.max_tool_rounds,
        max_completion_tokens=args.max_completion_tokens,
    )
    groups = load_groups(args.groups)
    if args.limit:
        groups = groups[:args.limit]
    print(f"Validated {len(groups)} groups; model={config.model}, reasoning_effort=high", flush=True)
    if args.dry_run:
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(args.output_dir / "manifest.json", {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "groups_file": str(args.groups.resolve()),
        "groups_sha256": hashlib.sha256(args.groups.read_bytes()).hexdigest(),
        "group_count": len(groups),
        "config": {item.name: getattr(config, item.name) for item in fields(config) if item.name != "api_key"},
    })
    analyzer = GeminiAPIAnalyzer(config)
    counts = {"ready": 0, "error": 0, "skipped": 0}
    try:
        for index, group in enumerate(groups, 1):
            # An index namespace prevents distinct source UIDs from colliding
            # after TeacherEvidenceBuilder sanitizes filenames.
            group_dir = args.output_dir / f"group-{index:06d}"
            group_dir.mkdir()
            _write_json(group_dir / "group.json", group.model_dump(mode="json"))
            builder = TeacherEvidenceBuilder(
                analyzer, NoDetectorFallback(),
                EvidenceBuilderConfig(output_dir=group_dir, reuse_cache=False, min_rollouts=1),
            )
            evidence = builder.build(group)
            _write_json(group_dir / "api_trace.json", analyzer.last_api_trace)
            counts[evidence.status] += 1
            print(f"[{index}/{len(groups)}] {evidence.status}; {len(evidence.crops)} selected crops", flush=True)
    finally:
        analyzer.close()
        _write_json(args.output_dir / "summary.json", counts)
    return 1 if counts["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
