#!/usr/bin/env python3
"""Read-only integrity verification for the DeepMath >=6 15% train subset."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(
    "/storage/yukaichengLab/wangxin/test_eiloeo/opd_data/"
    "deepmath_ge6_opd_train15pct_seed20260811"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ids(rows: list[dict]) -> list[int]:
    return [int(row["extra_info"]["source_index"]) for row in rows]


def difficulties(rows: list[dict]) -> dict[str, int]:
    counts = Counter(float(row["extra_info"]["difficulty"]) for row in rows)
    return {str(key): int(value) for key, value in sorted(counts.items())}


def main() -> None:
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "DEEPMATH_GE6_TRAIN15PCT_OK":
        raise RuntimeError("Invalid manifest status")
    for name, evidence in manifest["files"].items():
        path = ROOT / name
        if path.stat().st_size != evidence["bytes"] or sha256(path) != evidence["sha256"]:
            raise RuntimeError(f"File evidence mismatch: {name}")

    unique = pq.read_table(ROOT / "train_unique.parquet").to_pylist()
    train = pq.read_table(ROOT / "train.parquet").to_pylist()
    test = pq.read_table(ROOT / "test_1000.parquet").to_pylist()
    if (len(unique), len(train), len(test)) != (8_495, 8_512, 1_000):
        raise RuntimeError("Unexpected parquet row counts")
    unique_ids = ids(unique)
    test_ids = ids(test)
    if len(set(unique_ids)) != len(unique_ids):
        raise RuntimeError("Unique training parquet contains duplicate source ids")
    if set(unique_ids) & set(test_ids):
        raise RuntimeError("Training and held-out test sets overlap")
    padded = [row for row in train if bool(row["extra_info"]["padded_duplicate"])]
    if len(padded) != 17:
        raise RuntimeError("Unexpected padding-row count")
    if any(source_id not in set(unique_ids) for source_id in ids(padded)):
        raise RuntimeError("Padding row is not a duplicate of a sampled row")
    if difficulties(unique) != manifest["sampled_unique_difficulty_counts"]:
        raise RuntimeError("Unique difficulty counts mismatch")
    if difficulties(train) != manifest["padded_train_difficulty_counts"]:
        raise RuntimeError("Padded difficulty counts mismatch")
    selected = json.loads(
        (ROOT / "selected_source_indices.json").read_text(encoding="utf-8")
    )
    if selected["seed"] != 20260811 or selected["source_indices"] != unique_ids:
        raise RuntimeError("Selected source-index record mismatch")
    print(json.dumps({
        "status": "DEEPMATH_GE6_TRAIN15PCT_VERIFY_OK",
        "unique_rows": len(unique),
        "train_rows": len(train),
        "test_rows": len(test),
        "padding_rows": len(padded),
        "train_steps": len(train) // 64,
        "difficulty_counts": difficulties(unique),
        "train_sha256": sha256(ROOT / "train.parquet"),
        "test_sha256": sha256(ROOT / "test_1000.parquet"),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
