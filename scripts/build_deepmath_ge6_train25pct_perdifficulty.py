#!/usr/bin/env python3
"""Build a deterministic 25%-per-difficulty DeepMath >=6 train subset."""

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
    "deepmath_ge6_opd_train25pct_perdifficulty_seed20260823"
)
SOURCE_TRAIN = SOURCE_DIR / "train.parquet"
SOURCE_TEST = SOURCE_DIR / "test_1000.parquet"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
SEED = 20260823
REQUESTED_FRACTION = 0.25
SOURCE_UNIQUE_TRAIN_ROWS = 56_630
TARGET_UNIQUE_ROWS = 14_158
BATCH_SIZE = 64
EXPECTED_PADDED_ROWS = 14_208


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


def difficulty(row: dict[str, Any]) -> float:
    return float(row["extra_info"]["difficulty"])


def topic(row: dict[str, Any]) -> str:
    return str(row["extra_info"]["topic"])


def counts_by_difficulty(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(difficulty(row) for row in rows)
    return {str(key): int(value) for key, value in sorted(counts.items())}


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
        raise RuntimeError("Allocation exceeds a source stratum")
    return allocation


def main() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {OUTPUT_DIR}")
    source_manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    if source_manifest.get("status") != "DEEPMATH_GE6_OPD_DATA_OK":
        raise RuntimeError("Source manifest is not validated")

    source_table = pq.read_table(SOURCE_TRAIN)
    source_rows = source_table.to_pylist()
    unique_rows = [
        row for row in source_rows if not bool(row["extra_info"]["padded_duplicate"])
    ]
    if len(unique_rows) != SOURCE_UNIQUE_TRAIN_ROWS:
        raise RuntimeError(f"Unexpected unique source rows: {len(unique_rows)}")
    source_ids = [source_index(row) for row in unique_rows]
    if len(set(source_ids)) != len(source_ids):
        raise RuntimeError("Source training IDs are not unique")

    difficulty_groups: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in unique_rows:
        difficulty_groups[difficulty(row)].append(row)
    difficulty_population = {
        key: len(value) for key, value in difficulty_groups.items()
    }
    target = math.floor(len(unique_rows) * REQUESTED_FRACTION + 0.5)
    if target != TARGET_UNIQUE_ROWS:
        raise RuntimeError(f"Unexpected rounded 25% target: {target}")
    difficulty_allocation = largest_remainder(difficulty_population, target)

    selected: list[dict[str, Any]] = []
    topic_allocations: dict[tuple[float, str], int] = {}
    for level in sorted(difficulty_groups):
        by_topic: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in difficulty_groups[level]:
            by_topic[topic(row)].append(row)
        per_topic = largest_remainder(
            {key: len(value) for key, value in by_topic.items()},
            difficulty_allocation[level],
        )
        for topic_name in sorted(by_topic):
            candidates = sorted(by_topic[topic_name], key=source_index)
            take = per_topic[topic_name]
            topic_allocations[(level, topic_name)] = take
            if not take:
                continue
            digest = hashlib.sha256(
                f"{SEED}|{level!r}|{topic_name}".encode("utf-8")
            ).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:8], "big", signed=False))
            positions = np.sort(rng.choice(len(candidates), size=take, replace=False))
            selected.extend(deepcopy(candidates[int(position)]) for position in positions)
    selected.sort(key=source_index)
    selected_ids = [source_index(row) for row in selected]
    if len(selected) != TARGET_UNIQUE_ROWS or len(set(selected_ids)) != len(selected):
        raise RuntimeError("Selected training set is incomplete or duplicated")

    padding = (-len(selected)) % BATCH_SIZE
    padded_rows = deepcopy(selected)
    sample_groups: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        sample_groups[difficulty(row)].append(row)
    padding_allocation = largest_remainder(
        {key: len(value) for key, value in sample_groups.items()}, padding
    )
    for level in sorted(sample_groups):
        for row in sample_groups[level][: padding_allocation[level]]:
            duplicate = deepcopy(row)
            duplicate["extra_info"]["padded_duplicate"] = True
            padded_rows.append(duplicate)
    if len(padded_rows) != EXPECTED_PADDED_ROWS or len(padded_rows) % BATCH_SIZE:
        raise RuntimeError("Unexpected batch-aligned training size")

    test_rows = pq.read_table(SOURCE_TEST).to_pylist()
    test_ids = {source_index(row) for row in test_rows}
    if len(test_rows) != 1000 or set(selected_ids) & test_ids:
        raise RuntimeError("Held-out test size changed or train/test overlap exists")

    OUTPUT_DIR.mkdir(parents=True)
    atomic_parquet(OUTPUT_DIR / "train_unique.parquet", selected, source_table.schema)
    atomic_parquet(OUTPUT_DIR / "train.parquet", padded_rows, source_table.schema)
    shutil.copyfile(SOURCE_TEST, OUTPUT_DIR / "test_1000.parquet")
    atomic_json(
        OUTPUT_DIR / "selected_source_indices.json",
        {"seed": SEED, "source_indices": selected_ids},
    )

    parent_counts = counts_by_difficulty(unique_rows)
    sample_counts = counts_by_difficulty(selected)
    padded_counts = counts_by_difficulty(padded_rows)
    manifest: dict[str, Any] = {
        "status": "DEEPMATH_GE6_TRAIN25PCT_PER_DIFFICULTY_OK",
        "source_dir": str(SOURCE_DIR),
        "source_manifest_sha256": sha256(SOURCE_MANIFEST),
        "source_train_sha256": sha256(SOURCE_TRAIN),
        "source_test_sha256": sha256(SOURCE_TEST),
        "seed": SEED,
        "sampling": (
            "25% largest-remainder allocation per difficulty; full topic "
            "proportions preserved within each difficulty"
        ),
        "requested_fraction": REQUESTED_FRACTION,
        "source_unique_train_rows": SOURCE_UNIQUE_TRAIN_ROWS,
        "sampled_unique_rows": len(selected),
        "actual_unique_fraction": len(selected) / SOURCE_UNIQUE_TRAIN_ROWS,
        "batch_size": BATCH_SIZE,
        "padding_rows": padding,
        "padding_allocation_by_difficulty": {
            str(key): int(value) for key, value in sorted(padding_allocation.items())
        },
        "train_total_rows": len(padded_rows),
        "train_steps": len(padded_rows) // BATCH_SIZE,
        "test_rows": len(test_rows),
        "test_reused_unchanged": True,
        "source_strata": sum(
            len({topic(row) for row in rows}) for rows in difficulty_groups.values()
        ),
        "sampled_nonempty_strata": sum(value > 0 for value in topic_allocations.values()),
        "parent_difficulty_counts": parent_counts,
        "requested_difficulty_allocation": {
            str(key): int(value) for key, value in sorted(difficulty_allocation.items())
        },
        "sampled_unique_difficulty_counts": sample_counts,
        "padded_train_difficulty_counts": padded_counts,
        "files": {},
    }
    for name in (
        "train_unique.parquet",
        "train.parquet",
        "test_1000.parquet",
        "selected_source_indices.json",
    ):
        path = OUTPUT_DIR / name
        rows = pq.read_metadata(path).num_rows if path.suffix == ".parquet" else None
        manifest["files"][name] = {
            "bytes": path.stat().st_size,
            "rows": rows,
            "sha256": sha256(path),
        }
    if manifest["files"]["test_1000.parquet"]["sha256"] != sha256(SOURCE_TEST):
        raise RuntimeError("Held-out test is not byte-identical to source")
    atomic_json(OUTPUT_DIR / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    print("DEEPMATH_GE6_TRAIN25PCT_PER_DIFFICULTY_OK")


if __name__ == "__main__":
    main()
