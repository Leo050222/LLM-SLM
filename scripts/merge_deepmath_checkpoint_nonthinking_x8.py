#!/usr/bin/env python3
"""Strictly verify, merge, and summarize an eight-rollout checkpoint eval."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--full-data", type=Path, required=True)
    parser.add_argument("--expected-data-sha256", required=True)
    parser.add_argument("--expected-model", required=True)
    parser.add_argument("--expected-tokenizer", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--num-rollouts", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--max-tokens", type=int, default=40264)
    parser.add_argument("--max-model-len", type=int, default=40960)
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty output: {args.output_dir}")
    if sha256(args.full_data) != args.expected_data_sha256:
        raise RuntimeError("Full-data SHA mismatch")
    source_rows = pq.read_table(args.full_data).to_pylist()
    if len(source_rows) != 1000:
        raise RuntimeError(f"Expected 1000 source rows, got {len(source_rows)}")
    source: dict[int, dict[str, Any]] = {}
    for dataset_index, row in enumerate(source_rows):
        extra, reward = row["extra_info"], row["reward_model"]
        source[dataset_index] = {
            "problem_id": str(extra["problem_id"]),
            "problem": str(extra["question"]),
            "solution": str(reward["ground_truth"]),
            "difficulty": float(extra["difficulty"]),
            "topic": str(extra["topic"]),
            "source_index": int(extra["source_index"]),
        }

    rows: list[dict[str, Any]] = []
    elapsed = 0.0
    total_tokens_from_configs = 0
    for shard_index in range(args.num_shards):
        shard_dir = args.shard_root / f"shard_{shard_index}"
        config = read_json(shard_dir / "run_config.json")
        expected_config = {
            "status": "DEEPMATH_CHECKPOINT_NATIVE_NONTHINKING_X8_SHARD_OK",
            "student": args.expected_model,
            "tokenizer": args.expected_tokenizer,
            "shard_index": shard_index,
            "num_shards": args.num_shards,
            "num_prompts": 1000 // args.num_shards,
            "num_rollouts": (1000 // args.num_shards) * args.num_rollouts,
            "thinking_mode": False,
            "parameters_frozen": True,
            "optimizer_created": False,
            "backward_called": False,
        }
        for key, expected in expected_config.items():
            if config.get(key) != expected:
                raise RuntimeError(
                    f"{shard_dir}: {key}={config.get(key)!r}, expected {expected!r}"
                )
        sampling_expected = {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "max_tokens": args.max_tokens,
            "max_model_len": args.max_model_len,
            "n": args.num_rollouts,
            "seed": args.seed,
        }
        for key, expected in sampling_expected.items():
            if config["sampling"].get(key) != expected:
                raise RuntimeError(f"{shard_dir}: sampling.{key} mismatch")
        rollout_path = shard_dir / "rollouts.jsonl"
        if sha256(rollout_path) != config["output_sha256"]:
            raise RuntimeError(f"{shard_dir}: output SHA mismatch")
        shard_rows = read_jsonl(rollout_path)
        if len(shard_rows) != (1000 // args.num_shards) * args.num_rollouts:
            raise RuntimeError(f"{shard_dir}: rollout count mismatch")
        if any(int(row["dataset_index"]) % args.num_shards != shard_index for row in shard_rows):
            raise RuntimeError(f"{shard_dir}: modulo-shard membership mismatch")
        rows.extend(shard_rows)
        elapsed += float(config["elapsed_seconds_generate"])
        total_tokens_from_configs += int(config["total_response_tokens"])

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        dataset_index = int(row["dataset_index"])
        grouped[dataset_index].append(row)
        reference = source[dataset_index]
        for field, expected in reference.items():
            observed = row.get(field)
            if field == "difficulty":
                observed = float(observed)
            elif field == "source_index":
                observed = int(observed)
            else:
                observed = str(observed)
            if observed != expected:
                raise RuntimeError(f"index={dataset_index}: mismatch in {field}")
        response = str(row["response"])
        if "<think>" in response or "</think>" in response:
            raise RuntimeError(f"index={dataset_index}: thinking tag found")
        response_ids = row.get("response_token_ids")
        if not isinstance(response_ids, list) or len(response_ids) != int(row["response_tokens"]):
            raise RuntimeError(f"index={dataset_index}: response token mismatch")
        if int(row["response_tokens"]) > args.max_tokens:
            raise RuntimeError(f"index={dataset_index}: response exceeds cap")

    if set(grouped) != set(range(1000)):
        raise RuntimeError("Dataset indices are incomplete or extra")
    histogram = Counter()
    for dataset_index, group in grouped.items():
        group.sort(key=lambda row: int(row["replicate_index"]))
        if [int(row["replicate_index"]) for row in group] != list(range(args.num_rollouts)):
            raise RuntimeError(f"index={dataset_index}: replicate coverage mismatch")
        histogram[sum(bool(row["correct"]) for row in group)] += 1

    rows.sort(key=lambda row: (int(row["dataset_index"]), int(row["replicate_index"])))
    lengths = [int(row["response_tokens"]) for row in rows]
    correct = sum(bool(row["correct"]) for row in rows)
    total_tokens = sum(lengths)
    if total_tokens != total_tokens_from_configs:
        raise RuntimeError("Total response-token count mismatch")
    summary = {
        "status": "DEEPMATH_CHECKPOINT_NATIVE_NONTHINKING_X8_COMPLETE",
        "model_label": args.model_label,
        "model": args.expected_model,
        "tokenizer": args.expected_tokenizer,
        "data": str(args.full_data),
        "data_sha256": args.expected_data_sha256,
        "questions": 1000,
        "rollouts_per_question": args.num_rollouts,
        "rollouts": len(rows),
        "correct_rollouts": correct,
        "accuracy": correct / len(rows),
        "pass_at_8_count": 1000 - histogram[0],
        "pass_at_8": (1000 - histogram[0]) / 1000,
        "all_8_correct_count": histogram[8],
        "all_8_correct_rate": histogram[8] / 1000,
        "correct_count_histogram": {str(i): histogram[i] for i in range(9)},
        "total_response_tokens": total_tokens,
        "mean_response_tokens": total_tokens / len(rows),
        "max_response_tokens_observed": max(lengths),
        "finish_reason": dict(sorted(Counter(str(row["finish_reason"]) for row in rows).items())),
        "thinking_open_tag": 0,
        "thinking_close_tag": 0,
        "sampling": {
            "thinking": False,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "seed": args.seed,
            "n": args.num_rollouts,
            "max_response_tokens": args.max_tokens,
            "max_model_len": args.max_model_len,
        },
        "elapsed_seconds_generate_sum_shards": elapsed,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_jsonl(args.output_dir / "rollouts.jsonl", rows)
    atomic_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "NONTHINKING_X8_COMPLETE").touch()
    print(json.dumps(summary, sort_keys=True))
    print("DEEPMATH_CHECKPOINT_NATIVE_NONTHINKING_X8_COMPLETE")


if __name__ == "__main__":
    main()
