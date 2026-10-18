#!/usr/bin/env python3
"""Build a deterministic 15% difficulty/topic-stratified DeepMath train subset."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


SOURCE_DIR = Path(
    "/storage/yukaichengLab/wangxin/test_eiloeo/opd_data/deepmath_ge6_opd"
)
OUTPUT_DIR = Path(
    "/storage/yukaichengLab/wangxin/test_eiloeo/opd_data/"
    "deepmath_ge6_opd_train15pct_seed20260811"
)
SOURCE_TRAIN = SOURCE_DIR / "train.parquet"
SOURCE_TEST = SOURCE_DIR / "test_1000.parquet"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
SEED = 20260811
SOURCE_UNIQUE_TRAIN_ROWS = 56_630
TARGET_UNIQUE_ROWS = 8_495
BATCH_SIZE = 64
EXPECTED_PADDED_ROWS = 8_512


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_parquet(path: Path, rows: list[dict[str, Any]], schema: pa.Schema) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), temporary, compression="zstd")
    temporary.replace(path)


def source_index(row: dict[str, Any]) -> int:
    return int(row["extra_info"]["source_index"])


def stratum(row: dict[str, Any]) -> tuple[float, str]:
    extra = row["extra_info"]
    return float(extra["difficulty"]), str(extra["topic"])


def difficulty_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(float(row["extra_info"]["difficulty"]) for row in rows)
    return {str(key): int(value) for key, value in sorted(counts.items())}


def topic_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row["extra_info"]["topic"]) for row in rows)
    return {key: int(value) for key, value in sorted(counts.items())}


def largest_remainder(counts: dict[Any, int], total: int) -> dict[Any, int]:
    population = sum(counts.values())
    ideals = {key: value * total / population for key, value in counts.items()}
    allocation = {key: math.floor(value) for key, value in ideals.items()}
    remaining = total - sum(allocation.values())
    ranked = sorted(
        counts,
        key=lambda key: (-(ideals[key] - allocation[key]), repr(key)),
    )
    for key in ranked[:remaining]:
        allocation[key] += 1
    if sum(allocation.values()) != total:
        raise RuntimeError("Largest-remainder allocation did not reach target")
    if any(allocation[key] > counts[key] for key in counts):
        raise RuntimeError("A stratum allocation exceeds its population")
    return allocation


def main() -> None:
    source_manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    if source_manifest.get("status") != "DEEPMATH_GE6_OPD_DATA_OK":
        raise RuntimeError("Source manifest is not validated")

    source_table = pq.read_table(SOURCE_TRAIN)
    source_rows = source_table.to_pylist()
    unique_rows = [
        row for row in source_rows if not bool(row["extra_info"]["padded_duplicate"])
    ]
    if len(unique_rows) != SOURCE_UNIQUE_TRAIN_ROWS:
        raise RuntimeError(
            f"Expected {SOURCE_UNIQUE_TRAIN_ROWS} unique rows, found {len(unique_rows)}"
        )
    unique_ids = [source_index(row) for row in unique_rows]
    if len(set(unique_ids)) != len(unique_ids):
        raise RuntimeError("Source training rows are not unique by source_index")

    groups: dict[tuple[float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in unique_rows:
        groups[stratum(row)].append(row)
    difficulty_population = Counter(
        float(row["extra_info"]["difficulty"]) for row in unique_rows
    )
    difficulty_allocation = largest_remainder(
        dict(difficulty_population), TARGET_UNIQUE_ROWS
    )

    selected: list[dict[str, Any]] = []
    topic_allocations: dict[tuple[float, str], int] = {}
    for difficulty in sorted(difficulty_population):
        difficulty_groups = {
            topic: rows
            for (value, topic), rows in groups.items()
            if value == difficulty
        }
        per_topic = largest_remainder(
            {topic: len(rows) for topic, rows in difficulty_groups.items()},
            difficulty_allocation[difficulty],
        )
        for topic in sorted(difficulty_groups):
            rows = sorted(difficulty_groups[topic], key=source_index)
            take = per_topic[topic]
            topic_allocations[(difficulty, topic)] = take
            if take == 0:
                continue
            digest = hashlib.sha256(
                f"{SEED}|{difficulty!r}|{topic}".encode("utf-8")
            ).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:8], "big", signed=False))
            positions = np.sort(rng.choice(len(rows), size=take, replace=False))
            selected.extend(deepcopy(rows[int(position)]) for position in positions)
    selected.sort(key=source_index)

    if len(selected) != TARGET_UNIQUE_ROWS:
        raise RuntimeError(f"Expected {TARGET_UNIQUE_ROWS} sampled rows, found {len(selected)}")
    selected_ids = [source_index(row) for row in selected]
    if len(set(selected_ids)) != TARGET_UNIQUE_ROWS:
        raise RuntimeError("Sampled source indices are not unique")

    padded_rows = deepcopy(selected)
    padding = (-len(padded_rows)) % BATCH_SIZE
    for row in selected[:padding]:
        duplicate = deepcopy(row)
        duplicate["extra_info"]["padded_duplicate"] = True
        padded_rows.append(duplicate)
    if len(padded_rows) != EXPECTED_PADDED_ROWS or len(padded_rows) % BATCH_SIZE:
        raise RuntimeError("Unexpected batch-aligned training size")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    targets = [
        OUTPUT_DIR / "train_unique.parquet",
        OUTPUT_DIR / "train.parquet",
        OUTPUT_DIR / "test_1000.parquet",
        OUTPUT_DIR / "selected_source_indices.json",
        OUTPUT_DIR / "manifest.json",
    ]
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing outputs: {existing}")

    atomic_parquet(OUTPUT_DIR / "train_unique.parquet", selected, source_table.schema)
    atomic_parquet(OUTPUT_DIR / "train.parquet", padded_rows, source_table.schema)
    shutil.copyfile(SOURCE_TEST, OUTPUT_DIR / "test_1000.parquet")
    atomic_json(
        OUTPUT_DIR / "selected_source_indices.json",
        {"seed": SEED, "source_indices": selected_ids},
    )

    test_rows = pq.read_table(OUTPUT_DIR / "test_1000.parquet").to_pylist()
    if len(test_rows) != 1000:
        raise RuntimeError("Held-out test set row count changed")
    if sha256(OUTPUT_DIR / "test_1000.parquet") != sha256(SOURCE_TEST):
        raise RuntimeError("Held-out test file is not byte-identical to source")

    parent_difficulty = difficulty_counts(unique_rows)
    sample_difficulty = difficulty_counts(selected)
    padded_difficulty = difficulty_counts(padded_rows)
    maximum_difficulty_share_error_pp = max(
        abs(
            100 * sample_difficulty.get(key, 0) / TARGET_UNIQUE_ROWS
            - 100 * value / SOURCE_UNIQUE_TRAIN_ROWS
        )
        for key, value in parent_difficulty.items()
    )
    manifest = {
        "status": "DEEPMATH_GE6_TRAIN15PCT_OK",
        "source_dir": str(SOURCE_DIR),
        "source_manifest_sha256": sha256(SOURCE_MANIFEST),
        "source_train_sha256": sha256(SOURCE_TRAIN),
        "source_test_sha256": sha256(SOURCE_TEST),
        "seed": SEED,
        "sampling": (
            "hierarchical largest remainder: difficulty first, then full topic "
            "within each difficulty"
        ),
        "requested_fraction": 0.15,
        "source_unique_train_rows": SOURCE_UNIQUE_TRAIN_ROWS,
        "sampled_unique_rows": TARGET_UNIQUE_ROWS,
        "actual_unique_fraction": TARGET_UNIQUE_ROWS / SOURCE_UNIQUE_TRAIN_ROWS,
        "batch_size": BATCH_SIZE,
        "padding_rows": padding,
        "train_total_rows": len(padded_rows),
        "train_steps": len(padded_rows) // BATCH_SIZE,
        "test_rows": len(test_rows),
        "test_reused_unchanged": True,
        "source_strata": len(groups),
        "sampled_nonempty_strata": sum(
            value > 0 for value in topic_allocations.values()
        ),
        "source_topics": len(topic_counts(unique_rows)),
        "sampled_topics": len(topic_counts(selected)),
        "parent_difficulty_counts": parent_difficulty,
        "requested_difficulty_allocation": {
            str(key): int(value)
            for key, value in sorted(difficulty_allocation.items())
        },
        "sampled_unique_difficulty_counts": sample_difficulty,
        "padded_train_difficulty_counts": padded_difficulty,
        "maximum_difficulty_share_error_pp": maximum_difficulty_share_error_pp,
        "files": {},
    }
    for name in [
        "train_unique.parquet",
        "train.parquet",
        "test_1000.parquet",
        "selected_source_indices.json",
    ]:
        path = OUTPUT_DIR / name
        manifest["files"][name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    atomic_json(OUTPUT_DIR / "manifest.json", manifest)

    check = json.loads((OUTPUT_DIR / "manifest.json").read_text(encoding="utf-8"))
    if check != manifest:
        raise RuntimeError("Manifest round-trip mismatch")
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    print("DEEPMATH_GE6_TRAIN15PCT_OK")


if __name__ == "__main__":
    main()
