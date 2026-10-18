#!/usr/bin/env python3
"""Analyze aligned generation dumps from verl's internal validation path."""

import argparse
import json
import math
import re
from pathlib import Path


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def source(row: dict) -> str:
    return "gsm8k" if 'after "####"' in row["input"] else "math"


def strict_answer(row: dict):
    if source(row) == "gsm8k":
        matches = re.findall(r"#### (-?[0-9.,]+)", row["output"][-300:])
        return matches[-1].replace(",", "") if matches else None
    return "present" if "\\boxed" in row["output"] else None


def flexible_answer(row: dict):
    numbers = re.findall(r"-?[0-9.,]+", row["output"][-300:])
    for answer in reversed(numbers):
        if answer not in {"", "."}:
            return answer.replace(",", "")
    return None


def mcnemar_p(regressions: int, gains: int) -> float:
    n = regressions + gains
    if not n:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(regressions, gains) + 1)) / 2**n
    return min(1.0, 2 * tail)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--opd", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    models = {"base4": load(args.base), "opd4": load(args.opd), "base14": load(args.teacher)}
    if len({len(rows) for rows in models.values()}) != 1:
        raise RuntimeError("Generation dump lengths differ")
    for index in range(len(models["base4"])):
        inputs = {rows[index]["input"] for rows in models.values()}
        if len(inputs) != 1:
            raise RuntimeError(f"Inputs are not aligned at row {index}")

    report = {"scores": {}, "transitions": {}}
    for name, rows in models.items():
        report["scores"][name] = {}
        for dataset in ("gsm8k", "math"):
            subset = [row for row in rows if source(row) == dataset]
            item = {
                "correct": sum(row["score"] == 1.0 for row in subset),
                "total": len(subset),
                "missing_strict_answer": sum(strict_answer(row) is None for row in subset),
            }
            if dataset == "gsm8k":
                item["flexible_correct"] = sum(
                    flexible_answer(row) == str(row["gts"]).replace(",", "") for row in subset
                )
            report["scores"][name][dataset] = item

    for dataset in ("gsm8k", "math"):
        indices = [i for i, row in enumerate(models["base4"]) if source(row) == dataset]
        regressions = [
            i for i in indices if models["base4"][i]["score"] == 1.0 and models["opd4"][i]["score"] != 1.0
        ]
        gains = [
            i for i in indices if models["base4"][i]["score"] != 1.0 and models["opd4"][i]["score"] == 1.0
        ]
        item = {
            "regressions": len(regressions),
            "gains": len(gains),
            "net": len(gains) - len(regressions),
            "mcnemar_exact_p": mcnemar_p(len(regressions), len(gains)),
            "regressions_teacher_wrong": sum(models["base14"][i]["score"] != 1.0 for i in regressions),
            "regressions_opd_missing_strict": sum(strict_answer(models["opd4"][i]) is None for i in regressions),
        }
        if dataset == "gsm8k":
            item["regressions_opd_flexible_correct"] = sum(
                flexible_answer(models["opd4"][i]) == str(models["opd4"][i]["gts"]).replace(",", "")
                for i in regressions
            )
        report["transitions"][dataset] = item

    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
