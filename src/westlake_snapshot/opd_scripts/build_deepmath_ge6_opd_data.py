#!/usr/bin/env python3
"""Build a deterministic DeepMath difficulty>=6 OPD train/test split."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer


SOURCE_DIR = Path(
    "/storage/yukaichengLab/wangxin/test_eiloeo/opd_data/deepmath_103k/source/data"
)
OUTPUT_DIR = Path(
    "/storage/yukaichengLab/wangxin/test_eiloeo/opd_data/deepmath_ge6_opd"
)
STUDENT_MODEL = Path("/storage/yukaichengLab/share/model/Qwen3-1.7B")
TEACHER_MODEL = Path("/storage/yukaichengLab/share/model/Qwen3-14B")
REVISION = "5cf055d1fe3d7a2eb19719ac020211469736ae44"
SEED = 20260810
MIN_DIFFICULTY = 6.0
TEST_SIZE = 1000
TRAIN_BATCH_SIZE = 64
EXPECTED_SOURCE_ROWS = 103_022
EXPECTED_FILTERED_ROWS = 57_630
EXPECTED_MAX_PROMPT_TOKENS = 695
PREFIX = (
    "Solve the following math problem. Show your reasoning and put your "
    "final answer inside \\boxed{}.\n\n"
)


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


def atomic_parquet(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, temporary, compression="zstd")
    temporary.replace(path)


def largest_remainder(counts: pd.Series, total: int) -> pd.Series:
    ideal = counts.astype(float) * (total / int(counts.sum()))
    allocation = np.floor(ideal).astype("int64")
    allocation = pd.Series(allocation, index=counts.index, dtype="int64")
    remaining = total - int(allocation.sum())
    if remaining:
        remainders = (ideal - allocation).sort_values(ascending=False, kind="stable")
        for stratum in remainders.index[:remaining]:
            allocation.loc[stratum] += 1
    if int(allocation.sum()) != total or (allocation > counts).any():
        raise RuntimeError("Invalid largest-remainder allocation")
    return allocation


def tokenizer_fingerprint(model: Path) -> dict[str, str]:
    return {
        path.name: sha256(path)
        for path in sorted(model.glob("tokenizer*"))
        if path.is_file()
    }


def prompt_lengths(frame: pd.DataFrame, tokenizer: AutoTokenizer) -> np.ndarray:
    lengths: list[int] = []
    questions = frame["question"].astype(str).tolist()
    for start in range(0, len(questions), 512):
        conversations = [
            [{"role": "user", "content": PREFIX + question}]
            for question in questions[start : start + 512]
        ]
        encoded = tokenizer.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=True,
            return_dict=True,
        )
        token_ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
        lengths.extend(len(item) for item in token_ids)
    return np.asarray(lengths, dtype=np.int32)


def make_row(row: pd.Series, split: str, padded_duplicate: bool = False) -> dict:
    question = str(row["question"])
    answer = str(row["final_answer"])
    source_index = int(row["source_index"])
    return {
        "data_source": "deepmath_103k",
        "prompt": [{"role": "user", "content": PREFIX + question}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": {
            "split": split,
            "index": source_index,
            "source_index": source_index,
            "problem_id": f"deepmath-{source_index}",
            "answer": answer,
            "question": question,
            "difficulty": float(row["difficulty"]),
            "topic": str(row["topic"]),
            "dataset_revision": REVISION,
            "padded_duplicate": bool(padded_duplicate),
        },
    }


def main() -> None:
    shards = sorted(SOURCE_DIR.glob("train-*.parquet"))
    if len(shards) != 10:
        raise RuntimeError(f"Expected 10 source shards, found {len(shards)}")
    columns = ["question", "final_answer", "difficulty", "topic"]
    frames = [pq.read_table(path, columns=columns).to_pandas() for path in shards]
    source = pd.concat(frames, ignore_index=True)
    if len(source) != EXPECTED_SOURCE_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_SOURCE_ROWS} rows, found {len(source)}")
    if source[columns].isna().any().any():
        raise RuntimeError("Required source fields contain null values")
    source.insert(0, "source_index", np.arange(len(source), dtype=np.int64))
    source["difficulty"] = source["difficulty"].astype(float)
    source["topic"] = source["topic"].astype(str)
    filtered = source[source["difficulty"] >= MIN_DIFFICULTY].copy()
    if len(filtered) != EXPECTED_FILTERED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_FILTERED_ROWS} filtered rows, found {len(filtered)}"
        )

    counts = filtered.groupby(["difficulty", "topic"], sort=True).size()
    allocation = largest_remainder(counts, TEST_SIZE)
    test_indices: list[int] = []
    for (difficulty, topic), take in allocation.items():
        if int(take) == 0:
            continue
        group = filtered[
            (filtered["difficulty"] == difficulty) & (filtered["topic"] == topic)
        ]
        digest = hashlib.sha256(
            f"{SEED}|{difficulty!r}|{topic}".encode("utf-8")
        ).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "big", signed=False))
        positions = np.sort(rng.choice(len(group), size=int(take), replace=False))
        test_indices.extend(group.iloc[positions].index.tolist())

    test = filtered.loc[sorted(test_indices)].copy()
    train = filtered.drop(index=test.index).copy()
    train = train.sort_values("source_index", kind="stable").reset_index(drop=True)
    test = test.sort_values("source_index", kind="stable").reset_index(drop=True)
    if len(test) != TEST_SIZE or len(train) != EXPECTED_FILTERED_ROWS - TEST_SIZE:
        raise RuntimeError("Unexpected train/test split sizes")
    if set(train["source_index"]) & set(test["source_index"]):
        raise RuntimeError("Train/test source indices overlap")

    student_fp = tokenizer_fingerprint(STUDENT_MODEL)
    teacher_fp = tokenizer_fingerprint(TEACHER_MODEL)
    if student_fp != teacher_fp:
        raise RuntimeError("Student and teacher tokenizer files differ")
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL, local_files_only=True)
    lengths = prompt_lengths(filtered, tokenizer)
    if len(lengths) != len(filtered) or int(lengths.max()) != EXPECTED_MAX_PROMPT_TOKENS:
        raise RuntimeError(
            f"Unexpected prompt lengths: rows={len(lengths)} max={int(lengths.max())}"
        )
    filtered = filtered.copy()
    filtered["prompt_tokens"] = lengths

    train_rows = [make_row(row, "train") for _, row in train.iterrows()]
    padding = (-len(train_rows)) % TRAIN_BATCH_SIZE
    for row in train_rows[:padding]:
        duplicate = deepcopy(row)
        duplicate["extra_info"]["padded_duplicate"] = True
        train_rows.append(duplicate)
    test_rows = [make_row(row, "test") for _, row in test.iterrows()]

    longest_source_index = int(filtered.iloc[int(lengths.argmax())]["source_index"])
    smoke_candidates = filtered.sort_values(
        ["difficulty", "prompt_tokens", "source_index"],
        ascending=[False, False, True],
        kind="stable",
    )
    selected_source_indices = [longest_source_index]
    for source_index in smoke_candidates["source_index"].astype(int):
        if source_index not in selected_source_indices:
            selected_source_indices.append(source_index)
        if len(selected_source_indices) == 4:
            break
    smoke_frame = filtered.set_index("source_index").loc[selected_source_indices].reset_index()
    smoke_rows = [make_row(row, "smoke") for _, row in smoke_frame.iterrows()]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_path = OUTPUT_DIR / "train.parquet"
    test_path = OUTPUT_DIR / "test_1000.parquet"
    smoke_path = OUTPUT_DIR / "smoke_train.parquet"
    atomic_parquet(train_path, train_rows)
    atomic_parquet(test_path, test_rows)
    atomic_parquet(smoke_path, smoke_rows)

    if pq.read_metadata(train_path).num_rows != len(train_rows):
        raise RuntimeError("Train parquet row count mismatch")
    if pq.read_metadata(test_path).num_rows != len(test_rows):
        raise RuntimeError("Test parquet row count mismatch")
    if pq.read_metadata(smoke_path).num_rows != len(smoke_rows):
        raise RuntimeError("Smoke parquet row count mismatch")

    difficulty_counts = {
        str(float(key)): int(value)
        for key, value in filtered.groupby("difficulty").size().sort_index().items()
    }
    train_difficulty_counts = {
        str(float(key)): int(value)
        for key, value in train.groupby("difficulty").size().sort_index().items()
    }
    test_difficulty_counts = {
        str(float(key)): int(value)
        for key, value in test.groupby("difficulty").size().sort_index().items()
    }
    manifest = {
        "status": "DEEPMATH_GE6_OPD_DATA_OK",
        "dataset": "zwhe99/DeepMath-103K",
        "revision": REVISION,
        "seed": SEED,
        "min_difficulty": MIN_DIFFICULTY,
        "source_rows": len(source),
        "filtered_unique_rows": len(filtered),
        "train_unique_rows": len(train),
        "train_padding_rows": padding,
        "train_total_rows": len(train_rows),
        "train_batch_size": TRAIN_BATCH_SIZE,
        "train_steps": len(train_rows) // TRAIN_BATCH_SIZE,
        "test_rows": len(test_rows),
        "split": "difficulty x full topic proportional largest remainder",
        "source_strata": int(len(counts)),
        "test_strata": int((allocation > 0).sum()),
        "topics": int(filtered["topic"].nunique()),
        "difficulty_counts": difficulty_counts,
        "train_difficulty_counts": train_difficulty_counts,
        "test_difficulty_counts": test_difficulty_counts,
        "prompt_tokens": {
            "min": int(lengths.min()),
            "p50": float(np.percentile(lengths, 50)),
            "p90": float(np.percentile(lengths, 90)),
            "p95": float(np.percentile(lengths, 95)),
            "p99": float(np.percentile(lengths, 99)),
            "p999": float(np.percentile(lengths, 99.9)),
            "max": int(lengths.max()),
            "max_source_index": longest_source_index,
        },
        "tokenizer_files": student_fp,
        "files": {
            train_path.name: {"rows": len(train_rows), "sha256": sha256(train_path)},
            test_path.name: {"rows": len(test_rows), "sha256": sha256(test_path)},
            smoke_path.name: {"rows": len(smoke_rows), "sha256": sha256(smoke_path)},
        },
        "smoke_source_indices": selected_source_indices,
    }
    atomic_json(OUTPUT_DIR / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
