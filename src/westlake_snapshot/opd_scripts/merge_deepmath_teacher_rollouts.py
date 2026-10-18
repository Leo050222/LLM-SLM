#!/usr/bin/env python3
"""Merge four sharded Qwen3-14B DeepMath rollout files with protocol checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-max-tokens", type=int, required=True)
    parser.add_argument("--expected-max-model-len", type=int, required=True)
    parser.add_argument("--full-data", type=Path, required=True)
    args = parser.parse_args()

    shard_dirs = sorted(args.shard_root.glob("shard_*"))
    if len(shard_dirs) != 4:
        raise RuntimeError(f"Expected 4 teacher shards, found {len(shard_dirs)}")

    rows: list[dict[str, Any]] = []
    configs: list[dict[str, Any]] = []
    for shard_dir in shard_dirs:
        config = json.loads((shard_dir / "run_config.json").read_text(encoding="utf-8"))
        configs.append(config)
        if config["sampling"].get("max_tokens") != args.expected_max_tokens:
            raise RuntimeError(f"Bad max_tokens in {shard_dir}")
        if config.get("max_model_len") != args.expected_max_model_len:
            raise RuntimeError(f"Bad max_model_len in {shard_dir}")
        if config.get("thinking_mode") is not True:
            raise RuntimeError(f"Thinking mode disabled in {shard_dir}")
        rows.extend(read_jsonl(shard_dir / "rollouts.jsonl"))

    rows.sort(key=lambda row: int(row["dataset_index"]))
    indices = [int(row["dataset_index"]) for row in rows]
    if len(rows) != 1000 or indices != list(range(1000)):
        raise RuntimeError("Teacher rows are incomplete, duplicated, or out of order")
    problem_ids = [str(row["problem_id"]) for row in rows]
    if len(set(problem_ids)) != 1000:
        raise RuntimeError("Teacher problem IDs are not unique")

    config = dict(configs[0])
    config["data_path"] = str(args.full_data)
    config["num_prompts"] = 1000
    config["elapsed_seconds_generate_sum_shards"] = sum(
        float(item.get("elapsed_seconds_generate", 0.0)) for item in configs
    )
    config["merged_from"] = [str(path) for path in shard_dirs]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "rollouts.jsonl", rows)
    write_json(args.output_dir / "run_config.json", config)
    (args.output_dir / "MERGE_COMPLETE").touch()
    print("DEEPMATH_TEACHER_MERGE_COMPLETE rows=1000")


if __name__ == "__main__":
    main()
