#!/usr/bin/env python3
"""Merge sharded DeepMath eval/diagnostic artifacts and verify the protocol."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SEED = 20260809
EXPECTED_ROWS = 1000
EXPECTED_SHARDS = 4
EXPECTED_TOPK = 128
DEFAULT_EXPECTED_MAX_TOKENS = 4096
DEFAULT_EXPECTED_MAX_MODEL_LEN = 8192


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def describe(values: pd.Series) -> dict[str, float | int]:
    array = values.to_numpy(dtype=np.float64)
    finite = array[np.isfinite(array)]
    if not finite.size:
        return {"count": int(array.size), "finite_count": 0}
    return {
        "count": int(array.size),
        "finite_count": int(finite.size),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
        "sem": float(np.std(finite, ddof=1) / math.sqrt(finite.size))
        if finite.size > 1
        else 0.0,
        "p50": float(np.quantile(finite, 0.50)),
        "p90": float(np.quantile(finite, 0.90)),
        "p95": float(np.quantile(finite, 0.95)),
        "p99": float(np.quantile(finite, 0.99)),
        "max": float(np.max(finite)),
    }


def eval_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "correct": int(sum(bool(row["correct"]) for row in rows)),
        "total": len(rows),
        "accuracy": float(np.mean([bool(row["correct"]) for row in rows])),
        "response_tokens": describe(
            pd.Series([int(row["response_tokens"]) for row in rows], dtype=float)
        ),
        "finish_reason": dict(Counter(str(row["finish_reason"]) for row in rows)),
        "score_method": dict(Counter(str(row["score_method"]) for row in rows)),
        "thinking_open_tag": int(sum("<think>" in row["response"] for row in rows)),
        "thinking_close_tag": int(sum("</think>" in row["response"] for row in rows)),
    }


def grouped_accuracy(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    frame = pd.DataFrame(
        [{key: row[key], "correct": bool(row["correct"])} for row in rows]
    )
    result: dict[str, Any] = {}
    for value, group in frame.groupby(key, observed=True, sort=True):
        result[str(value)] = {
            "correct": int(group["correct"].sum()),
            "total": int(len(group)),
            "accuracy": float(group["correct"].mean()),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostic-root", type=Path, required=True)
    parser.add_argument("--teacher-output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--student-key", default="qwen3_4b")
    parser.add_argument(
        "--expected-student",
        default="/storage/yukaichengLab/share/model/Qwen3-4B",
    )
    parser.add_argument(
        "--expected-max-tokens", type=int, default=DEFAULT_EXPECTED_MAX_TOKENS
    )
    parser.add_argument(
        "--expected-max-model-len", type=int, default=DEFAULT_EXPECTED_MAX_MODEL_LEN
    )
    args = parser.parse_args()

    shard_dirs = sorted(args.diagnostic_root.glob("shard_*"))
    if len(shard_dirs) != EXPECTED_SHARDS:
        raise RuntimeError(f"Expected {EXPECTED_SHARDS} diagnostic shards, found {len(shard_dirs)}")

    student_rows: list[dict[str, Any]] = []
    token_paths: list[Path] = []
    for shard_dir in shard_dirs:
        config = json.loads((shard_dir / "run_config.json").read_text(encoding="utf-8"))
        sampling = config["sampling"]
        expected = {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "max_tokens": args.expected_max_tokens,
            "seed": SEED,
        }
        for key, value in expected.items():
            if sampling.get(key) != value:
                raise RuntimeError(f"Bad sampling {key} in {shard_dir}: {sampling.get(key)!r}")
        if not config.get("thinking_mode") or config.get("topk") != EXPECTED_TOPK:
            raise RuntimeError(f"Bad thinking/top-k config in {shard_dir}")
        if config.get("max_model_len") != args.expected_max_model_len:
            raise RuntimeError(
                f"Bad max_model_len in {shard_dir}: {config.get('max_model_len')!r}"
            )
        if config.get("student") != args.expected_student:
            raise RuntimeError(
                f"Unexpected student in {shard_dir}: {config.get('student')!r}"
            )
        if config.get("student_parameters_require_grad") is not False:
            raise RuntimeError(f"Student was not frozen in {shard_dir}")
        student_rows.extend(read_jsonl(shard_dir / "rollouts.jsonl"))
        token_paths.extend(sorted((shard_dir / "token_shards").glob("example_*.parquet")))

    teacher_rows = read_jsonl(args.teacher_output / "rollouts.jsonl")
    for label, rows in (("student", student_rows), ("teacher", teacher_rows)):
        if len(rows) != EXPECTED_ROWS:
            raise RuntimeError(f"Expected {EXPECTED_ROWS} {label} rows, found {len(rows)}")
        indices = [int(row["dataset_index"]) for row in rows]
        if len(set(indices)) != EXPECTED_ROWS or set(indices) != set(range(EXPECTED_ROWS)):
            raise RuntimeError(f"{label} dataset indices are incomplete or duplicated")
    student_rows.sort(key=lambda row: int(row["dataset_index"]))
    teacher_rows.sort(key=lambda row: int(row["dataset_index"]))
    for student, teacher in zip(student_rows, teacher_rows, strict=True):
        if student["problem_id"] != teacher["problem_id"]:
            raise RuntimeError(f"Unaligned problem at {student['dataset_index']}")

    teacher_config = json.loads(
        (args.teacher_output / "run_config.json").read_text(encoding="utf-8")
    )
    for key, value in {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "max_tokens": args.expected_max_tokens,
        "seed": SEED,
    }.items():
        if teacher_config["sampling"].get(key) != value:
            raise RuntimeError(f"Bad teacher sampling {key}")
    if teacher_config.get("thinking_mode") is not True:
        raise RuntimeError("Teacher thinking mode is not recorded as enabled")
    if teacher_config.get("max_model_len") != args.expected_max_model_len:
        raise RuntimeError("Bad teacher max_model_len")

    if len(token_paths) != EXPECTED_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_ROWS} token shards, found {len(token_paths)}")
    tokens = pd.concat((pd.read_parquet(path) for path in token_paths), ignore_index=True)
    expected_tokens = sum(int(row["response_tokens"]) for row in student_rows)
    if len(tokens) != expected_tokens:
        raise RuntimeError(f"Token count mismatch: {len(tokens)} != {expected_tokens}")
    if tokens[["student_chosen_logprob", "teacher_chosen_logprob"]].isna().any().any():
        raise RuntimeError("Chosen-token log-probabilities contain NaN")

    metric_columns = [
        "chosen_logprob_gap",
        "abs_chosen_gap",
        "sampled_reverse_kl_k1",
        "sampled_reverse_kl_k3",
        "sampled_reverse_kl_k3_clipped20",
        "raw_forward_kl_top128",
        "effective_forward_kl_top128",
        "teacher_mass_top128",
        "student_mass_on_teacher_top128",
        "topk_overlap",
        "top1_agreement",
        "student_entropy",
        "teacher_entropy_lower_bound",
    ]
    per_rollout = (
        tokens.groupby(
            ["dataset_index", "problem_id", "difficulty", "topic", "correct"],
            observed=True,
        )[metric_columns]
        .mean()
        .reset_index()
    )
    lengths = tokens.groupby("dataset_index", observed=True).size().rename("response_tokens")
    per_rollout = per_rollout.join(lengths, on="dataset_index")

    student_correct = {int(row["dataset_index"]) for row in student_rows if row["correct"]}
    teacher_correct = {int(row["dataset_index"]) for row in teacher_rows if row["correct"]}
    paired = {
        "both_correct": len(student_correct & teacher_correct),
        f"only_{args.student_key}_correct": len(student_correct - teacher_correct),
        "only_14b_correct": len(teacher_correct - student_correct),
        "both_wrong": EXPECTED_ROWS - len(student_correct | teacher_correct),
    }
    summary: dict[str, Any] = {
        "protocol": {
            "dataset": "zwhe99/DeepMath-103K",
            "sample_rows": EXPECTED_ROWS,
            "stratification": "exact difficulty x full hierarchical topic",
            "sampling": {
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "max_response_tokens": args.expected_max_tokens,
                "max_model_len": args.expected_max_model_len,
                "n": 1,
                "seed": SEED,
                "thinking_mode": True,
            },
            "diagnostic": (
                f"{args.student_key}-sampled actions scored by frozen "
                f"{args.student_key} and 14B; top-k=128"
            ),
            "reverse_kl_note": (
                "k1 and k3 are Monte Carlo estimators of KL(student || teacher) "
                "under student-sampled actions; they are not exact full-vocabulary KL"
            ),
        },
        args.student_key: eval_summary(student_rows),
        "qwen3_14b": eval_summary(teacher_rows),
        "paired_correctness": paired,
        "token_diagnostics": {
            "num_tokens": int(len(tokens)),
            "token_weighted": {column: describe(tokens[column]) for column in metric_columns},
            "rollout_weighted": {
                column: describe(per_rollout[column]) for column in metric_columns
            },
            "by_correctness": {
                str(bool(correct)).lower(): {
                    "rollouts": int(len(group)),
                    "sampled_reverse_kl_k1": describe(group["sampled_reverse_kl_k1"]),
                    "sampled_reverse_kl_k3": describe(group["sampled_reverse_kl_k3"]),
                    "top1_agreement": describe(group["top1_agreement"]),
                    "topk_overlap": describe(group["topk_overlap"]),
                }
                for correct, group in per_rollout.groupby("correct", observed=True)
            },
        },
        "accuracy_by_difficulty": {
            args.student_key: grouped_accuracy(student_rows, "difficulty"),
            "qwen3_14b": grouped_accuracy(teacher_rows, "difficulty"),
        },
        "accuracy_by_topic": {
            args.student_key: grouped_accuracy(student_rows, "topic"),
            "qwen3_14b": grouped_accuracy(teacher_rows, "topic"),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / f"{args.student_key}_rollouts.jsonl", student_rows)
    write_jsonl(args.output_dir / "qwen3_14b_rollouts.jsonl", teacher_rows)
    tokens.to_parquet(args.output_dir / "tokens.parquet", index=False, compression="zstd")
    per_rollout.to_parquet(
        args.output_dir / "per_rollout_diagnostics.parquet", index=False, compression="zstd"
    )
    write_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "MERGE_COMPLETE").touch()
    print("DEEPMATH_MERGE_COMPLETE", json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
