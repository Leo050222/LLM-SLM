#!/usr/bin/env python3
"""Classify format-free evaluation failures as truncation or real answer errors."""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from score_format_free_final_answer import (
    GSM,
    MATH,
    extract_primary,
    format_free_score,
    last_number_score,
    load,
    source,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--opd", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--max-response-length", type=int, default=2048)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--records-output", type=Path, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    models = {
        "qwen3_4b_base": load(args.base),
        "qwen3_4b_opd": load(args.opd),
        "qwen3_14b_base": load(args.teacher),
    }

    report = {
        "truncation_definition": f"retokenized generated response length >= {args.max_response_length}",
        "max_response_length": args.max_response_length,
        "models": {},
    }
    all_records = []
    for model_name, rows in models.items():
        report["models"][model_name] = {}
        for dataset in (GSM, MATH):
            failures = []
            for row_index, row in enumerate(rows):
                if source(row) != dataset or format_free_score(row):
                    continue
                output_tokens = len(tokenizer.encode(row["output"], add_special_tokens=False))
                truncated = output_tokens >= args.max_response_length
                candidate = extract_primary(row)
                if truncated:
                    category = "truncated"
                elif candidate is not None:
                    category = "completed_wrong_answer"
                else:
                    category = "completed_no_final_answer"
                record = {
                    "model": model_name,
                    "dataset": dataset,
                    "row": row_index,
                    "ground_truth": row["gts"],
                    "candidate": candidate,
                    "output_tokens": output_tokens,
                    "closed_think": "</think>" in row["output"],
                    "category": category,
                }
                if dataset == GSM:
                    record["last_number_correct"] = last_number_score(row)
                failures.append(record)
                all_records.append(record)

            total = len(failures)
            counts = {
                category: sum(row["category"] == category for row in failures)
                for category in ("truncated", "completed_wrong_answer", "completed_no_final_answer")
            }
            report["models"][model_name][dataset] = {
                "failures": total,
                **counts,
                **{f"{key}_pct": value / total if total else 0.0 for key, value in counts.items()},
                "not_closed_think": sum(not row["closed_think"] for row in failures),
                "truncated_with_last_number_correct": sum(
                    row["category"] == "truncated" and row.get("last_number_correct", False) for row in failures
                ),
            }

    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with args.records_output.open("w", encoding="utf-8") as handle:
        for record in all_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
