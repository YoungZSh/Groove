#!/usr/bin/env python3
"""Run the external Qwen Analyzer with DINO and PaddleOCR on saved rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mmcot_opsd.analyzer import OpenAIAnalyzerConfig, OpenAICompatibleAnalyzer
from mmcot_opsd.schemas import GroupRollout, Rollout


def _question_from_input(value: str) -> str:
    question = value
    if question.startswith("user\n\n"):
        question = question[len("user\n\n") :]
    marker = "\nInspect the image carefully"
    return question.split(marker, 1)[0].strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", type=Path, required=True, help="one saved rollout JSONL group")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--uid", default="analyzer-tool-probe")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.rollouts.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError("rollout JSONL is empty")
    group = GroupRollout(
        uid=args.uid,
        question=_question_from_input(str(rows[0]["input"])),
        image_path=args.image.resolve(),
        rollouts=[
            Rollout(
                rollout_id=index,
                completion=str(row["output"]),
                predicted_label=row.get("predicted_label"),
                reward=float(row["score"]),
            )
            for index, row in enumerate(rows)
        ],
    )
    analyzer = OpenAICompatibleAnalyzer(OpenAIAnalyzerConfig.from_env())
    focus = analyzer.analyze(group)
    result = {
        "uid": group.uid,
        "question": group.question,
        "focus": focus.model_dump(mode="json"),
        "tool_trace": analyzer.last_tool_trace,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
