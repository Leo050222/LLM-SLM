#!/usr/bin/env python3
"""Compare aligned 2048- and 4096-token format-free evaluation records."""

import argparse
import json
from pathlib import Path


def load_records(path: Path) -> dict[tuple[str, int], dict]:
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        records[(row["model"], row["row"])] = row
    return records


def load_failures(path: Path) -> dict[tuple[str, int], dict]:
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        records[(row["model"], row["row"])] = row
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--short-records", type=Path, required=True)
    parser.add_argument("--long-records", type=Path, required=True)
    parser.add_argument("--short-failures", type=Path, required=True)
    parser.add_argument("--short-length", type=int, default=2048)
    parser.add_argument("--long-length", type=int, default=4096)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    short = load_records(args.short_records)
    long = load_records(args.long_records)
    short_failures = load_failures(args.short_failures)
    if short.keys() != long.keys():
        raise RuntimeError("2048 and 4096 record keys differ")

    model_names = sorted({key[0] for key in short})
    report = {
        "short_max_response_length": args.short_length,
        "long_max_response_length": args.long_length,
        "models": {},
    }
    for model in model_names:
        report["models"][model] = {}
        for dataset in ("gsm8k", "math"):
            keys = [key for key, row in short.items() if key[0] == model and row["source"] == dataset]
            gains = [key for key in keys if not short[key]["format_free_correct"] and long[key]["format_free_correct"]]
            regressions = [key for key in keys if short[key]["format_free_correct"] and not long[key]["format_free_correct"]]
            short_truncated = [
                key for key in keys if key in short_failures and short_failures[key]["category"] == "truncated"
            ]
            recovered_truncated = [key for key in short_truncated if long[key]["format_free_correct"]]
            report["models"][model][dataset] = {
                "short_correct": sum(short[key]["format_free_correct"] for key in keys),
                "long_correct": sum(long[key]["format_free_correct"] for key in keys),
                "net": len(gains) - len(regressions),
                "wrong_to_correct": len(gains),
                "correct_to_wrong": len(regressions),
                "short_truncated_failures": len(short_truncated),
                "short_truncated_recovered_at_long_length": len(recovered_truncated),
                "short_truncated_recovery_rate": len(recovered_truncated) / len(short_truncated) if short_truncated else 0,
            }

    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
