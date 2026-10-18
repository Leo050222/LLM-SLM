#!/usr/bin/env python3
"""Export a small, readable sample of validated anchor/bridge records."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--student-rollout-root", type=Path, required=True)
    parser.add_argument(
        "--tokenizer",
        default="/storage/yukaichengLab/share/model/Qwen3-14B",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def read_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Manifest line {line_number} is not an object")
            rows.append(value)
    return rows


def compact_anchor(anchor: dict[str, Any]) -> dict[str, Any]:
    selected = anchor["selected_candidate"]
    judgment = anchor.get("judge", {}).get("judgment", {})
    readable_alternatives = [
        {
            "token": token.get("token_text"),
            "probability": token.get("probability"),
        }
        for token in selected.get("teacher_top_tokens", [])[:5]
    ]
    return {
        "response_position": int(anchor["anchor_response_position"]),
        "student_token": anchor["student_candidate_token_text"],
        "absolute_logprob_gap": float(anchor["absolute_logprob_gap"]),
        "gap_direction": anchor["gap_direction"],
        "student_probability": selected.get("student_probability"),
        "teacher_probability_on_student_token": selected.get(
            "teacher_probability_on_student_token"
        ),
        "teacher_top_alternatives": readable_alternatives,
        "judge": {
            "select_anchor": judgment.get("select_anchor"),
            "label": judgment.get("label"),
            "confidence": judgment.get("confidence"),
            "reason": judgment.get("reason"),
            "student_direction": judgment.get("student_direction"),
            "teacher_possible_direction": judgment.get(
                "teacher_possible_direction"
            ),
        },
    }


def compact_rollouts(bundle: dict[str, Any]) -> list[str]:
    rollouts = bundle.get("rollouts")
    if not isinstance(rollouts, list) or len(rollouts) != 8:
        raise ValueError(
            f"Trace {bundle.get('dataset_index')} does not contain 8 rollouts"
        )
    result: list[str] = []
    for rollout in rollouts:
        token_ids = rollout.get("bridge_token_ids")
        if not isinstance(token_ids, list) or len(token_ids) != 256:
            raise ValueError(
                f"Trace {bundle.get('dataset_index')} rollout "
                f"{rollout.get('rollout_index')} is not 256 tokens"
            )
        if rollout.get("first_token_differs_from_student") is not True:
            raise ValueError("First-token constraint is not satisfied")
        result.append(rollout["bridge_text"])
    return result


def load_student_rollouts(
    root: Path, dataset_indices: set[int]
) -> dict[int, dict[str, Any]]:
    found: dict[int, dict[str, Any]] = {}
    for shard in sorted(root.glob("shard_*/rollouts.jsonl")):
        with shard.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                index = int(row["dataset_index"])
                if index in dataset_indices:
                    found[index] = row
        if len(found) == len(dataset_indices):
            break
    missing = sorted(dataset_indices - found.keys())
    if missing:
        raise ValueError(f"Missing student rollouts for dataset indices: {missing}")
    return found


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")

    from transformers import AutoTokenizer

    manifest = read_manifest(args.manifest)
    if len(manifest) < args.num_samples:
        raise ValueError(
            f"Requested {args.num_samples} samples from {len(manifest)} rows"
        )

    rng = random.Random(args.seed)
    sampled = rng.sample(manifest, args.num_samples)
    sampled.sort(key=lambda row: int(row["dataset_index"]))
    student_rows = load_student_rollouts(
        args.student_rollout_root,
        {int(row["dataset_index"]) for row in sampled},
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        local_files_only=True,
        trust_remote_code=True,
    )

    records: list[dict[str, Any]] = []
    for manifest_row in sampled:
        anchor_path = Path(manifest_row["anchor_artifact"])
        bridge_path = Path(manifest_row["bridge_artifact"])
        anchor = read_json(anchor_path)
        bundle = read_json(bridge_path)
        dataset_index = int(bundle["dataset_index"])
        student_row = student_rows[dataset_index]
        prefix_ids = [int(token_id) for token_id in anchor["anchor_prefix_token_ids"]]
        response_ids = [int(token_id) for token_id in student_row["response_token_ids"]]
        anchor_position = int(anchor["anchor_response_position"])
        if not 0 <= anchor_position < len(response_ids):
            raise ValueError(
                f"Trace {dataset_index} has invalid anchor position {anchor_position}"
            )
        suffix_ids = response_ids[anchor_position + 1 :]
        records.append(
            {
                "dataset_index": dataset_index,
                "problem_id": bundle["problem_id"],
                "difficulty": student_row.get("difficulty"),
                "topic": student_row.get("topic"),
                "problem": student_row.get("problem"),
                "prefix": tokenizer.decode(
                    prefix_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ),
                "anchor": compact_anchor(anchor),
                "original_student_trace_suffix": tokenizer.decode(
                    suffix_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ),
                "teacher_rollouts": compact_rollouts(bundle),
            }
        )

    output = {
        "schema_version": 1,
        "description": (
            "Readable deterministic preview. Each record shows the text before "
            "the anchor, the selected student anchor token, the remainder of "
            "the original student trace after that token, and eight teacher "
            "bridge rollouts. Token IDs and low-level generation fields are "
            "intentionally omitted."
        ),
        "source_manifest": str(args.manifest.resolve()),
        "sample_seed": args.seed,
        "sample_count": len(records),
        "samples": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(
        f"BRIDGE_SAMPLE_EXPORT_OK output={args.output} "
        f"samples={len(records)} seed={args.seed}"
    )


if __name__ == "__main__":
    main()
