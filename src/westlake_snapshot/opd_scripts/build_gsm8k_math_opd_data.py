#!/usr/bin/env python3
"""Build a schema-normalized, batch-aligned GSM8K+MATH OPD dataset."""

from pathlib import Path

from datasets import Dataset, load_dataset


ROOT = Path("/storage/yukaichengLab/wangxin/test_eiloeo/opd_data")
OUTPUT = ROOT / "gsm8k_math"
BATCH_SIZE = 64
VAL_PER_DATASET = 256
SEED = 20260806


def load_parquet(path: Path) -> Dataset:
    return load_dataset("parquet", data_files=str(path), split="train")


def normalize(row: dict, source: str) -> dict:
    extra = row.get("extra_info") or {}
    return {
        "data_source": row["data_source"],
        "prompt": row["prompt"],
        "ability": row["ability"],
        "reward_model": row["reward_model"],
        "extra_info": {
            "split": extra.get("split"),
            "index": extra.get("index"),
            "answer": extra.get("answer"),
            "question": extra.get("question"),
            "level": row.get("level"),
            "type": row.get("type"),
            "original_dataset": source,
            "padded_duplicate": False,
        },
    }


def normalized_rows(dataset: Dataset, source: str) -> list[dict]:
    return [normalize(row, source) for row in dataset]


def main() -> None:
    gsm_train = load_parquet(ROOT / "gsm8k" / "train.parquet")
    gsm_test = load_parquet(ROOT / "gsm8k" / "test.parquet")
    math_train = load_parquet(ROOT / "math" / "train.parquet")
    math_test = load_parquet(ROOT / "math" / "test.parquet")

    unique_train = normalized_rows(gsm_train, "gsm8k") + normalized_rows(math_train, "math")
    padding = (-len(unique_train)) % BATCH_SIZE
    padded_rows = []
    for row in unique_train[:padding]:
        duplicate = {
            **row,
            "extra_info": {**row["extra_info"], "padded_duplicate": True},
        }
        padded_rows.append(duplicate)
    train_rows = unique_train + padded_rows

    gsm_val = gsm_test.shuffle(seed=SEED).select(range(VAL_PER_DATASET))
    math_val = math_test.shuffle(seed=SEED).select(range(VAL_PER_DATASET))
    val_rows = normalized_rows(gsm_val, "gsm8k") + normalized_rows(math_val, "math")
    full_test_rows = normalized_rows(gsm_test, "gsm8k") + normalized_rows(math_test, "math")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(train_rows).to_parquet(OUTPUT / "train.parquet")
    Dataset.from_list(val_rows).to_parquet(OUTPUT / "validation_512.parquet")
    Dataset.from_list(full_test_rows).to_parquet(OUTPUT / "test_full.parquet")

    print(
        "DATASET_OK",
        f"unique_train={len(unique_train)}",
        f"padding={padding}",
        f"train={len(train_rows)}",
        f"validation={len(val_rows)}",
        f"full_test={len(full_test_rows)}",
        f"batch_size={BATCH_SIZE}",
        f"steps={len(train_rows) // BATCH_SIZE}",
    )


if __name__ == "__main__":
    main()
