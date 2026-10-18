#!/usr/bin/env python3
"""Read-only integrity verification for the DeepMath-103K 1000-item pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


EXPECTED_SAMPLE_SHA256 = (
    "1bb2f3f4df8aac7ab802d7b34276b27799e49ecf4bf6587c62b838a8f8ac4fb6"
)
EXPECTED_ROWS = 1000
EXPECTED_INDICES = set(range(EXPECTED_ROWS))
EXPECTED_SAMPLING = {
    "max_tokens": 4096,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": -1,
    "n": 1,
    "seed": 20260809,
}
METRIC_COLUMNS = [
    "student_chosen_logprob",
    "teacher_chosen_logprob",
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def walk_numbers(value: Any):
    if isinstance(value, dict):
        for child in value.values():
            yield from walk_numbers(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_numbers(child)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield float(value)


def validate_rollouts(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    assert len(rows) == EXPECTED_ROWS, (label, len(rows))
    indices = [int(row["dataset_index"]) for row in rows]
    assert len(set(indices)) == EXPECTED_ROWS, f"{label}: duplicate indices"
    assert set(indices) == EXPECTED_INDICES, f"{label}: incomplete index set"
    assert all(
        int(row["response_tokens"]) == len(row["response_token_ids"])
        for row in rows
    ), f"{label}: response token mismatch"
    assert sum("<think>" in row["response"] for row in rows) == EXPECTED_ROWS, (
        f"{label}: thinking mode evidence missing"
    )
    return {
        "rows": len(rows),
        "unique_indices": len(set(indices)),
        "correct": sum(bool(row["correct"]) for row in rows),
        "response_tokens": sum(int(row["response_tokens"]) for row in rows),
        "finish_reason": dict(Counter(str(row["finish_reason"]) for row in rows)),
        "thinking_open": sum("<think>" in row["response"] for row in rows),
        "thinking_close": sum("</think>" in row["response"] for row in rows),
    }


def validate_config(path: Path, expected_student: str, diagnostic: bool) -> None:
    config = json.loads(path.read_text(encoding="utf-8"))
    assert config["student"] == expected_student, (path, config["student"])
    assert config["thinking_mode"] is True, path
    for key, value in EXPECTED_SAMPLING.items():
        assert config["sampling"].get(key) == value, (
            path,
            key,
            config["sampling"].get(key),
        )
    assert config["sampling"].get("do_sample") is True, path
    if diagnostic:
        assert config["teacher"].endswith("/Qwen3-14B"), path
        assert config["topk"] == 128, path
        assert config["parameters_frozen"] is True, path
        assert config["optimizer_created"] is False, path
        assert config["backward_called"] is False, path
        assert config["student_parameters_require_grad"] is False, path
        assert config["tokenizers_equal"] is True, path


def validate_parquet_finite(path: Path) -> int:
    parquet = pq.ParquetFile(path)
    rows = 0
    for batch in parquet.iter_batches(batch_size=131072, columns=METRIC_COLUMNS):
        rows += batch.num_rows
        for column in batch.columns:
            values = column.to_numpy(zero_copy_only=False)
            assert np.isfinite(values).all(), f"non-finite values in {path}"
    return rows


def key(row: dict[str, Any]) -> tuple[int, str]:
    return int(row["dataset_index"]), str(row["problem_id"])


def metadata(rows: list[dict[str, Any]]) -> dict[tuple[int, str], tuple[Any, ...]]:
    return {
        key(row): (
            row["problem"],
            row["difficulty"],
            row["topic"],
            row["source_index"],
        )
        for row in rows
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "/storage/yukaichengLab/wangxin/test_eiloeo/"
            "opd_eval_deepmath/full_seed20260809"
        ),
    )
    parser.add_argument(
        "--sample",
        type=Path,
        default=Path(
            "/storage/yukaichengLab/wangxin/test_eiloeo/"
            "opd_data/deepmath_103k/sample_1000.parquet"
        ),
    )
    args = parser.parse_args()
    root = args.root

    actual_sha256 = sha256(args.sample)
    assert actual_sha256 == EXPECTED_SAMPLE_SHA256, actual_sha256

    rollout_paths = {
        "qwen3_1_7b": root / "merged_qwen3_1_7b/qwen3_1_7b_rollouts.jsonl",
        "qwen3_4b": root / "merged/qwen3_4b_rollouts.jsonl",
        "qwen3_14b": root / "merged/qwen3_14b_rollouts.jsonl",
    }
    rows = {label: read_jsonl(path) for label, path in rollout_paths.items()}
    model_summaries = {
        label: validate_rollouts(model_rows, label)
        for label, model_rows in rows.items()
    }

    key_sets = [{key(row) for row in model_rows} for model_rows in rows.values()]
    assert key_sets[0] == key_sets[1] == key_sets[2], "model problem sets differ"
    metadata_maps = [metadata(model_rows) for model_rows in rows.values()]
    assert metadata_maps[0] == metadata_maps[1] == metadata_maps[2], (
        "model problem metadata differ"
    )

    for path in sorted((root / "diagnostic").glob("shard_*/run_config.json")):
        validate_config(path, "/storage/yukaichengLab/share/model/Qwen3-4B", True)
    for path in sorted(
        (root / "diagnostic_qwen3_1_7b").glob("shard_*/run_config.json")
    ):
        validate_config(path, "/storage/yukaichengLab/share/model/Qwen3-1.7B", True)
    validate_config(
        root / "qwen3_14b/run_config.json",
        "/storage/yukaichengLab/share/model/Qwen3-14B",
        False,
    )

    artifact_counts: dict[str, dict[str, int]] = {}
    for label, directory in {
        "qwen3_1_7b": root / "diagnostic_qwen3_1_7b",
        "qwen3_4b": root / "diagnostic",
    }.items():
        topk = list(directory.glob("shard_*/teacher_topk/example_*.npz"))
        token_shards = list(directory.glob("shard_*/token_shards/example_*.parquet"))
        assert len(topk) == EXPECTED_ROWS, (label, "teacher_topk", len(topk))
        assert len(token_shards) == EXPECTED_ROWS, (
            label,
            "token_shards",
            len(token_shards),
        )
        artifact_counts[label] = {
            "teacher_topk": len(topk),
            "token_shards": len(token_shards),
        }

    merged_summaries = [
        json.loads((root / "merged/summary.json").read_text(encoding="utf-8")),
        json.loads(
            (root / "merged_qwen3_1_7b/summary.json").read_text(encoding="utf-8")
        ),
    ]
    for summary in merged_summaries:
        assert all(math.isfinite(value) for value in walk_numbers(summary))

    parquet_rows = {
        "qwen3_1_7b": validate_parquet_finite(
            root / "merged_qwen3_1_7b/tokens.parquet"
        ),
        "qwen3_4b": validate_parquet_finite(root / "merged/tokens.parquet"),
    }
    assert parquet_rows["qwen3_1_7b"] == model_summaries["qwen3_1_7b"][
        "response_tokens"
    ]
    assert parquet_rows["qwen3_4b"] == model_summaries["qwen3_4b"][
        "response_tokens"
    ]

    correct_sets = {
        label: {
            int(row["dataset_index"]) for row in model_rows if bool(row["correct"])
        }
        for label, model_rows in rows.items()
    }
    triple_overlap = {
        f"1.7B={small},4B={medium},14B={teacher}": sum(
            ((index in correct_sets["qwen3_1_7b"]) == bool(small))
            and ((index in correct_sets["qwen3_4b"]) == bool(medium))
            and ((index in correct_sets["qwen3_14b"]) == bool(teacher))
            for index in EXPECTED_INDICES
        )
        for small in (0, 1)
        for medium in (0, 1)
        for teacher in (0, 1)
    }

    output = {
        "status": "DEEPMATH_TIER3_EVIDENCE_OK",
        "sample_sha256": actual_sha256,
        "same_1000_problem_ids": True,
        "same_problem_metadata": True,
        "models": model_summaries,
        "diagnostic_artifacts": artifact_counts,
        "finite_token_rows": parquet_rows,
        "triple_correctness": triple_overlap,
    }
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
