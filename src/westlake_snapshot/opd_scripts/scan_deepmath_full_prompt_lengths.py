#!/usr/bin/env python3
"""Stream DeepMath-103K and report Qwen3 thinking-prompt token lengths."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer


SOURCE = Path(
    "/storage/yukaichengLab/wangxin/test_eiloeo/opd_data/deepmath_103k/source/data"
)
MODEL = Path("/storage/yukaichengLab/share/model/Qwen3-1.7B")
PREFIX = (
    "Solve the following math problem. Show your reasoning and put your "
    "final answer inside \\boxed{}.\n\n"
)


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    lengths: list[int] = []
    for shard in sorted(SOURCE.glob("train-*.parquet")):
        questions = pq.read_table(shard, columns=["question"])["question"].to_pylist()
        for question in questions:
            encoded = tokenizer.apply_chat_template(
                [{"role": "user", "content": PREFIX + str(question)}],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            token_ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
            lengths.append(len(token_ids))

    values = np.asarray(lengths, dtype=np.int32)
    quantiles = np.percentile(values, [50, 90, 95, 99, 99.9])
    print(
        "DEEPMATH_FULL_PROMPT_LENGTHS",
        f"rows={len(values)}",
        f"min={int(values.min())}",
        f"p50={quantiles[0]:.1f}",
        f"p90={quantiles[1]:.1f}",
        f"p95={quantiles[2]:.1f}",
        f"p99={quantiles[3]:.1f}",
        f"p999={quantiles[4]:.1f}",
        f"max={int(values.max())}",
        f"max_source_index={int(values.argmax())}",
        f"over512={int((values > 512).sum())}",
        f"over1024={int((values > 1024).sum())}",
        f"over2048={int((values > 2048).sum())}",
    )


if __name__ == "__main__":
    main()
