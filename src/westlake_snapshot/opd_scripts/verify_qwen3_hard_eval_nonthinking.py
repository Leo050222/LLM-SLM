#!/usr/bin/env python3
"""Read-only evidence check for the completed Qwen3 non-thinking hard eval."""

from __future__ import annotations

import glob
import json
import os
import subprocess
from collections import Counter
from pathlib import Path


ROOT = Path(
    "/storage/yukaichengLab/wangxin/test_eiloeo/"
    "opd_eval_hard/results/nonthinking_full"
)
EXPECTED_ROWS = {
    "aime24_default.jsonl": 30,
    "aime25_default.jsonl": 30,
    "hmmt25_default.jsonl": 30,
    "olympiad_bench_OE_TO_maths_en_COMP.jsonl": 674,
    "gpqa_diamond_default.jsonl": 198,
    "live_code_bench_default.jsonl": 175,
}
EXPECTED_CORRECT = {
    "qwen3_4b": {
        "aime24_default.jsonl": 9,
        "aime25_default.jsonl": 6,
        "hmmt25_default.jsonl": 5,
        "olympiad_bench_OE_TO_maths_en_COMP.jsonl": 318,
        "gpqa_diamond_default.jsonl": 82,
        "live_code_bench_default.jsonl": 44,
    },
    "qwen3_14b": {
        "aime24_default.jsonl": 9,
        "aime25_default.jsonl": 8,
        "hmmt25_default.jsonl": 3,
        "olympiad_bench_OE_TO_maths_en_COMP.jsonl": 344,
        "gpqa_diamond_default.jsonl": 97,
        "live_code_bench_default.jsonl": 51,
    },
}
SPECS = {
    "qwen3_4b": "Qwen3-4B-NonThinking",
    "qwen3_14b": "Qwen3-14B-NonThinking",
}


def nonempty_reasoning_values(value):
    if isinstance(value, dict):
        for key, nested in value.items():
            if key.lower() in {"reasoning", "reasoning_content"} and nested not in (
                None,
                "",
                [],
            ):
                yield nested
            yield from nonempty_reasoning_values(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from nonempty_reasoning_values(nested)


def score_value(row: dict) -> float:
    values = row["sample_score"]["score"]["value"]
    return float(values.get("acc_pass@1", values.get("acc", next(iter(values.values())))))


def main() -> None:
    accounting = subprocess.check_output(
        [
            "/soft/slurm/bin/sacct",
            "-j",
            "86036,86069",
            "--format=JobID,State,ExitCode",
            "-n",
            "-P",
        ],
        text=True,
    )
    assert "86036_1|COMPLETED|0:0" in accounting
    assert "86069|COMPLETED|0:0" in accounting

    total_rows = 0
    output_tokens = {}
    finish_reasons = {}
    for tag, model_name in SPECS.items():
        model_root = ROOT / tag
        reports = list((model_root / "reports" / model_name).glob("*.json"))
        assert len(reports) == 6

        prediction_root = model_root / "predictions" / model_name
        review_root = model_root / "reviews" / model_name
        model_output_tokens = 0
        model_reasons = Counter()
        for filename, expected_rows in EXPECTED_ROWS.items():
            prediction_path = prediction_root / filename
            review_path = review_root / filename
            predictions = [json.loads(line) for line in prediction_path.open()]
            reviews = [json.loads(line) for line in review_path.open()]
            assert len(predictions) == expected_rows
            assert len(reviews) == expected_rows
            assert sum(score_value(row) for row in reviews) == EXPECTED_CORRECT[tag][filename]

            for row in predictions:
                serialized = json.dumps(row, ensure_ascii=False)
                assert "<think>" not in serialized and "</think>" not in serialized
                assert not list(nonempty_reasoning_values(row))
                model_output = row["model_output"]
                model_output_tokens += int((model_output.get("usage") or {}).get("output_tokens") or 0)
                choice = model_output["choices"][0]
                model_reasons[choice.get("finish_reason") or choice.get("stop_reason")] += 1

            total_rows += len(predictions)

        output_tokens[tag] = model_output_tokens
        finish_reasons[tag] = dict(model_reasons)

    assert total_rows == 2274
    assert output_tokens == {"qwen3_4b": 2_079_396, "qwen3_14b": 1_538_786}
    assert finish_reasons == {
        "qwen3_4b": {"stop": 1120, "max_tokens": 16, "model_length": 1},
        "qwen3_14b": {"stop": 1134, "max_tokens": 2, "model_length": 1},
    }
    print("TIER3_NONTHINKING_EVAL_EVIDENCE_OK")


if __name__ == "__main__":
    main()
