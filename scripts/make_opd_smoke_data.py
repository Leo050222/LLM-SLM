#!/usr/bin/env python3
"""Create a deterministic GSM8K-shaped dataset for an OPD smoke test."""

from pathlib import Path

from datasets import Dataset


OUTPUT_DIR = Path("/storage/yukaichengLab/wangxin/test_eiloeo/opd_data/gsm8k_smoke")


def make_example(index: int, split: str) -> dict:
    left = 7 + (index * 13) % 89
    right = 3 + (index * 17) % 47
    if index % 2 == 0:
        question = (
            f"A box contains {left} red marbles and {right} blue marbles. "
            "How many marbles are in the box?"
        )
        answer = left + right
    else:
        multiplier = 2 + index % 8
        question = (
            f"There are {left} shelves and each shelf holds {multiplier} books. "
            "How many books are there in total?"
        )
        answer = left * multiplier

    question += ' Think step by step and output the final answer after "####".'
    return {
        "data_source": "openai/gsm8k",
        "prompt": [{"role": "user", "content": question}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": str(answer)},
        "extra_info": {"split": split, "index": index, "answer": f"#### {answer}", "question": question},
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Dataset.from_list([make_example(i, "train") for i in range(128)]).to_parquet(OUTPUT_DIR / "train.parquet")
    Dataset.from_list([make_example(i + 10_000, "test") for i in range(32)]).to_parquet(
        OUTPUT_DIR / "test.parquet"
    )
    print(f"Wrote smoke data to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
