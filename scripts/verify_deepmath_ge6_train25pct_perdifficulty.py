#!/usr/bin/env python3
"""Read-only verification for the 25%-per-difficulty DeepMath >=6 subset."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


SOURCE = Path(
    "/storage/yukaichengLab/wangxin/test_eiloeo/opd_data/deepmath_ge6_opd"
)
ROOT = Path(
    "/storage/yukaichengLab/wangxin/test_eiloeo/opd_data/"
    "deepmath_ge6_opd_train25pct_perdifficulty_seed20260823"
)
EXPECTED_ALLOCATION = {
    "6.0": 4295,
    "6.5": 1904,
    "7.0": 2102,
    "7.5": 1657,
    "8.0": 2872,
    "8.5": 981,
    "9.0": 333,
    "9.5": 13,
    "10.0": 1,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_index(row: dict[str, Any]) -> int:
    return int(row["extra_info"]["source_index"])


def difficulty(row: dict[str, Any]) -> float:
    return float(row["extra_info"]["difficulty"])


def topic(row: dict[str, Any]) -> str:
    return str(row["extra_info"]["topic"])


def difficulty_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(difficulty(row) for row in rows)
    return {str(key): int(value) for key, value in sorted(counts.items())}


def largest_remainder(counts: dict[Any, int], total: int) -> dict[Any, int]:
    population = sum(counts.values())
    ideals = {key: value * total / population for key, value in counts.items()}
    allocation = {key: math.floor(value) for key, value in ideals.items()}
    remaining = total - sum(allocation.values())
    ranked = sorted(counts, key=lambda key: (-(ideals[key] - allocation[key]), repr(key)))
    for key in ranked[:remaining]:
        allocation[key] += 1
    return allocation


def main() -> None:
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "DEEPMATH_GE6_TRAIN25PCT_PER_DIFFICULTY_OK":
        raise RuntimeError("Invalid manifest status")
    for name, evidence in manifest["files"].items():
        path = ROOT / name
        if path.stat().st_size != evidence["bytes"] or sha256(path) != evidence["sha256"]:
            raise RuntimeError(f"File evidence mismatch: {name}")

    source_train = pq.read_table(SOURCE / "train.parquet").to_pylist()
    source_unique = [
        row for row in source_train if not bool(row["extra_info"]["padded_duplicate"])
    ]
    unique = pq.read_table(ROOT / "train_unique.parquet").to_pylist()
    train = pq.read_table(ROOT / "train.parquet").to_pylist()
    test = pq.read_table(ROOT / "test_1000.parquet").to_pylist()
    if (len(source_unique), len(unique), len(train), len(test)) != (
        56_630,
        14_158,
        14_208,
        1_000,
    ):
        raise RuntimeError("Unexpected parquet row counts")

    source_by_id = {source_index(row): row for row in source_unique}
    unique_ids = [source_index(row) for row in unique]
    test_ids = [source_index(row) for row in test]
    if len(set(unique_ids)) != len(unique_ids) or not set(unique_ids) <= set(source_by_id):
        raise RuntimeError("Selected IDs are duplicated or absent from source")
    if set(unique_ids) & set(test_ids):
        raise RuntimeError("Training and held-out test sets overlap")
    for row in unique:
        if row != source_by_id[source_index(row)]:
            raise RuntimeError("Selected row content differs from canonical source")

    if sha256(ROOT / "test_1000.parquet") != sha256(SOURCE / "test_1000.parquet"):
        raise RuntimeError("Held-out test is not byte-identical to canonical test")
    if difficulty_counts(unique) != EXPECTED_ALLOCATION:
        raise RuntimeError("Per-difficulty 25% allocation mismatch")
    if manifest["requested_difficulty_allocation"] != EXPECTED_ALLOCATION:
        raise RuntimeError("Manifest allocation mismatch")

    source_groups: dict[float, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    sample_groups: dict[float, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in source_unique:
        source_groups[difficulty(row)][topic(row)] += 1
    for row in unique:
        sample_groups[difficulty(row)][topic(row)] += 1
    for level, counts in source_groups.items():
        expected = largest_remainder(dict(counts), EXPECTED_ALLOCATION[str(level)])
        expected_nonzero = {key: value for key, value in expected.items() if value}
        if dict(sample_groups[level]) != expected_nonzero:
            raise RuntimeError(f"Topic allocation mismatch at difficulty {level}")

    padding = [row for row in train if bool(row["extra_info"]["padded_duplicate"])]
    if len(padding) != 50 or any(source_index(row) not in set(unique_ids) for row in padding):
        raise RuntimeError("Padding provenance mismatch")
    nonpadding = [row for row in train if not bool(row["extra_info"]["padded_duplicate"])]
    if [source_index(row) for row in nonpadding] != unique_ids:
        raise RuntimeError("Non-padding train rows differ from unique train rows")
    selected = json.loads((ROOT / "selected_source_indices.json").read_text(encoding="utf-8"))
    if selected.get("seed") != 20260823 or selected.get("source_indices") != unique_ids:
        raise RuntimeError("Selected source-index record mismatch")

    summary = {
        "status": "DEEPMATH_GE6_TRAIN25PCT_PER_DIFFICULTY_VERIFY_OK",
        "unique_rows": len(unique),
        "train_rows": len(train),
        "test_rows": len(test),
        "padding_rows": len(padding),
        "train_steps": len(train) // 64,
        "difficulty_counts": difficulty_counts(unique),
        "train_unique_sha256": sha256(ROOT / "train_unique.parquet"),
        "train_sha256": sha256(ROOT / "train.parquet"),
        "test_sha256": sha256(ROOT / "test_1000.parquet"),
    }
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
