#!/usr/bin/env python3
"""Strictly verify every trace in the DeepMath train-25% non-thinking rollout."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


EXPECTED_ROWS = 14_208
EXPECTED_PADDING = 50
EXPECTED_SOURCE_SHA256 = (
    "fdcb58732d22f5ab27d111f07c8746836d983c5d15d1d800970154ea1f3cfc91"
)
EXPECTED_MODEL = "/storage/yukaichengLab/share/model/Qwen3-1.7B"
EXPECTED_TOKENIZER = "/storage/yukaichengLab/share/model/Qwen3-1.7B"
NUM_SHARDS = 16
ROWS_PER_SHARD = 888
SEED = 20260811
MAX_TOKENS = 40_264
MAX_MODEL_LEN = 40_960


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


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def user_prompt(problem: str) -> str:
    return (
        "Solve the following math problem. Show your reasoning and put your "
        "final answer inside \\boxed{}.\n\n" + problem
    )


def reference(index: int, row: dict[str, Any]) -> dict[str, Any]:
    extra = row["extra_info"]
    reward = row["reward_model"]
    return {
        "dataset_index": index,
        "problem_id": str(extra["problem_id"]),
        "problem": str(extra["question"]),
        "solution": str(reward["ground_truth"]),
        "difficulty": float(extra["difficulty"]),
        "topic": str(extra["topic"]),
        "source_index": int(extra["source_index"]),
        "padded_duplicate": bool(extra["padded_duplicate"]),
    }


def quantiles(values: list[int]) -> dict[str, float]:
    ordered = sorted(values)
    result: dict[str, float] = {}
    for name, fraction in (("p50", 0.50), ("p75", 0.75), ("p90", 0.90), ("p95", 0.95), ("p99", 0.99)):
        position = fraction * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        weight = position - lower
        result[name] = ordered[lower] * (1.0 - weight) + ordered[upper] * weight
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    if (args.output_root / "ROLLOUT_COMPLETE").exists():
        raise FileExistsError("Refusing to overwrite completed verification")
    if sha256(args.source) != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("Source SHA mismatch")
    prepare = read_json(args.output_root / "prepare_manifest.json")
    expected_prepare = {
        "status": "DEEPMATH_TRAIN25_NATIVE_NONTHINKING_PREPARED",
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "rows": EXPECTED_ROWS,
        "padding_rows": EXPECTED_PADDING,
        "num_shards": NUM_SHARDS,
        "rows_per_shard": ROWS_PER_SHARD,
    }
    for key, expected in expected_prepare.items():
        if prepare.get(key) != expected:
            raise RuntimeError(f"prepare_manifest {key} mismatch")
    source_rows = pq.read_table(args.source).to_pylist()
    if len(source_rows) != EXPECTED_ROWS:
        raise RuntimeError("Unexpected source row count")
    references = [reference(index, row) for index, row in enumerate(source_rows)]

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        EXPECTED_TOKENIZER, local_files_only=True, trust_remote_code=False
    )
    all_rows: list[dict[str, Any]] = []
    shard_summaries: list[dict[str, Any]] = []
    for shard_index in range(NUM_SHARDS):
        input_path = args.output_root / "inputs" / f"shard_{shard_index:02d}.parquet"
        output_dir = args.output_root / "shards" / f"shard_{shard_index:02d}"
        if not (output_dir / "SHARD_COMPLETE").is_file():
            raise RuntimeError(f"Shard {shard_index} lacks completion marker")
        config = read_json(output_dir / "run_config.json")
        manifest_shard = prepare["shards"][shard_index]
        expected_config = {
            "status": "DEEPMATH_TRAIN25_NATIVE_NONTHINKING_SHARD_OK",
            "input_sha256": manifest_shard["sha256"],
            "model": EXPECTED_MODEL,
            "tokenizer": EXPECTED_TOKENIZER,
            "shard_index": shard_index,
            "num_shards": NUM_SHARDS,
            "num_prompts": ROWS_PER_SHARD,
            "thinking_mode": False,
            "parameters_frozen": True,
            "optimizer_created": False,
            "backward_called": False,
        }
        for key, expected in expected_config.items():
            if config.get(key) != expected:
                raise RuntimeError(f"Shard {shard_index}: config {key} mismatch")
        sampling_expected = {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "max_tokens": MAX_TOKENS,
            "max_model_len": MAX_MODEL_LEN,
            "n": 1,
            "seed": SEED,
        }
        for key, expected in sampling_expected.items():
            if config["sampling"].get(key) != expected:
                raise RuntimeError(f"Shard {shard_index}: sampling {key} mismatch")
        if sha256(input_path) != manifest_shard["sha256"]:
            raise RuntimeError(f"Shard {shard_index}: input SHA mismatch")
        rollout_path = output_dir / "rollouts.jsonl"
        if sha256(rollout_path) != config["output_sha256"]:
            raise RuntimeError(f"Shard {shard_index}: rollout SHA mismatch")
        rows = read_jsonl(rollout_path)
        if len(rows) != ROWS_PER_SHARD:
            raise RuntimeError(f"Shard {shard_index}: row count mismatch")
        if any(int(row["dataset_index"]) % NUM_SHARDS != shard_index for row in rows):
            raise RuntimeError(f"Shard {shard_index}: membership mismatch")
        if [int(row["dataset_index"]) for row in rows] != sorted(
            int(row["dataset_index"]) for row in rows
        ):
            raise RuntimeError(f"Shard {shard_index}: trace order mismatch")
        for row in rows:
            index = int(row["dataset_index"])
            expected = references[index]
            for field, expected_value in expected.items():
                observed = row.get(field)
                if field in {"dataset_index", "source_index"}:
                    observed = int(observed)
                elif field == "difficulty":
                    observed = float(observed)
                elif field == "padded_duplicate":
                    observed = bool(observed)
                else:
                    observed = str(observed)
                if observed != expected_value:
                    raise RuntimeError(f"index={index}: {field} mismatch")
            response = str(row["response"])
            response_ids = row.get("response_token_ids")
            prompt_ids = row.get("prompt_token_ids")
            if not isinstance(response_ids, list) or not all(
                isinstance(value, int) for value in response_ids
            ):
                raise RuntimeError(f"index={index}: invalid response_token_ids")
            if not isinstance(prompt_ids, list) or not prompt_ids:
                raise RuntimeError(f"index={index}: invalid prompt_token_ids")
            if len(response_ids) != int(row["response_tokens"]):
                raise RuntimeError(f"index={index}: response length mismatch")
            if len(response_ids) > MAX_TOKENS:
                raise RuntimeError(f"index={index}: response exceeds cap")
            if tokenizer.decode(response_ids, skip_special_tokens=False) != response:
                raise RuntimeError(f"index={index}: response decode mismatch")
            if "<think>" in response or "</think>" in response:
                raise RuntimeError(f"index={index}: thinking tag found")
            prompt_text = user_prompt(expected["problem"])
            if str(row["prompt_text"]) != prompt_text:
                raise RuntimeError(f"index={index}: prompt text mismatch")
            encoded = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_text}],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            expected_prompt_ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
            if prompt_ids != list(expected_prompt_ids):
                raise RuntimeError(f"index={index}: prompt token mismatch")
        all_rows.extend(rows)
        shard_summaries.append(
            {
                "shard_index": shard_index,
                "rows": len(rows),
                "padding_rows": sum(bool(row["padded_duplicate"]) for row in rows),
                "rollout_sha256": config["output_sha256"],
                "elapsed_seconds_generate": float(config["elapsed_seconds_generate"]),
                "total_response_tokens": sum(int(row["response_tokens"]) for row in rows),
                "node": config.get("node"),
                "vllm_port": config.get("vllm_port"),
            }
        )

    all_rows.sort(key=lambda row: int(row["dataset_index"]))
    if [int(row["dataset_index"]) for row in all_rows] != list(range(EXPECTED_ROWS)):
        raise RuntimeError("Global trace coverage is incomplete or duplicated")
    if sum(bool(row["padded_duplicate"]) for row in all_rows) != EXPECTED_PADDING:
        raise RuntimeError("Global padding count mismatch")
    lengths = [int(row["response_tokens"]) for row in all_rows]
    difficulty_groups: dict[float, list[int]] = defaultdict(list)
    for row in all_rows:
        difficulty_groups[float(row["difficulty"])].append(int(row["response_tokens"]))
    summary = {
        "status": "DEEPMATH_TRAIN25_NATIVE_NONTHINKING_ROLLOUT_OK",
        "source": str(args.source),
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "model": EXPECTED_MODEL,
        "tokenizer": EXPECTED_TOKENIZER,
        "rows": len(all_rows),
        "padding_rows": EXPECTED_PADDING,
        "unique_rows": EXPECTED_ROWS - EXPECTED_PADDING,
        "num_shards": NUM_SHARDS,
        "sampling": {
            "thinking": False,
            **sampling_expected,
        },
        "total_response_tokens": sum(lengths),
        "mean_response_tokens": sum(lengths) / len(lengths),
        "min_response_tokens": min(lengths),
        "max_response_tokens_observed": max(lengths),
        "response_token_quantiles": quantiles(lengths),
        "finish_reason": dict(sorted(Counter(str(row["finish_reason"]) for row in all_rows).items())),
        "by_difficulty": {
            str(level): {
                "rows": len(group),
                "mean_response_tokens": sum(group) / len(group),
                "response_token_quantiles": quantiles(group),
            }
            for level, group in sorted(difficulty_groups.items())
        },
        "shards": shard_summaries,
    }
    atomic_json(args.output_root / "summary.json", summary)
    (args.output_root / "ROLLOUT_COMPLETE").touch()
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    print("DEEPMATH_TRAIN25_NATIVE_NONTHINKING_ROLLOUT_OK")


if __name__ == "__main__":
    main()
