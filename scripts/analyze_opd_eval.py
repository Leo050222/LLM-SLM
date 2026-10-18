#!/usr/bin/env python3
"""Compare base student, OPD student, and teacher per-example evaluations."""

import argparse
import json
import math
import re
from pathlib import Path


GSM = "openai/gsm8k"
MATH = "DigitalLearningGmbH/MATH-lighteval"


def gsm_flexible_answer(response: str):
    numbers = re.findall(r"-?[0-9.,]+", response[-300:])
    for answer in reversed(numbers):
        if answer not in {"", "."}:
            return answer.replace(",", "")
    return None


def load(path: Path) -> dict[tuple[str, int], dict]:
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        records[(row["data_source"], row["index"])] = row
    return records


def exact_mcnemar_p(regressions: int, gains: int) -> float:
    discordant = regressions + gains
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(regressions, gains) + 1)) / (2**discordant)
    return min(1.0, 2 * tail)


def dataset_summary(records: dict, source: str) -> dict:
    rows = [row for (row_source, _), row in records.items() if row_source == source]
    summary = {
        "correct": sum(row["score"] == 1.0 for row in rows),
        "total": len(rows),
        "accuracy": sum(row["score"] == 1.0 for row in rows) / len(rows),
        "missing_strict_answer": sum(row["extracted_answer"] is None for row in rows),
        "length_finished": sum(row["finish_reason"] == "length" for row in rows),
        "mean_output_tokens": sum(row["output_tokens"] for row in rows) / len(rows),
    }
    if source == GSM:
        summary["flexible_correct"] = sum(
            gsm_flexible_answer(row["response"]) == str(row["ground_truth"]).replace(",", "")
            for row in rows
        )
        summary["flexible_accuracy"] = summary["flexible_correct"] / len(rows)
    return summary


def transitions(base: dict, opd: dict, teacher: dict, source: str) -> dict:
    keys = sorted(key for key in base if key[0] == source)
    regressions = [key for key in keys if base[key]["score"] == 1.0 and opd[key]["score"] != 1.0]
    gains = [key for key in keys if base[key]["score"] != 1.0 and opd[key]["score"] == 1.0]
    stable_correct = [key for key in keys if base[key]["score"] == 1.0 and opd[key]["score"] == 1.0]
    stable_wrong = [key for key in keys if base[key]["score"] != 1.0 and opd[key]["score"] != 1.0]

    result = {
        "stable_correct": len(stable_correct),
        "regressions": len(regressions),
        "gains": len(gains),
        "stable_wrong": len(stable_wrong),
        "net": len(gains) - len(regressions),
        "mcnemar_exact_p": exact_mcnemar_p(len(regressions), len(gains)),
        "regression_details": {
            "teacher_correct": sum(teacher[key]["score"] == 1.0 for key in regressions),
            "teacher_wrong": sum(teacher[key]["score"] != 1.0 for key in regressions),
            "opd_missing_strict_answer": sum(opd[key]["extracted_answer"] is None for key in regressions),
            "opd_length_finished": sum(opd[key]["finish_reason"] == "length" for key in regressions),
            "base_length_finished": sum(base[key]["finish_reason"] == "length" for key in regressions),
            "teacher_length_finished": sum(teacher[key]["finish_reason"] == "length" for key in regressions),
            "opd_matches_teacher_extracted_answer": sum(
                opd[key]["extracted_answer"] is not None
                and opd[key]["extracted_answer"] == teacher[key]["extracted_answer"]
                for key in regressions
            ),
            "indices": [key[1] for key in regressions],
        },
        "gain_details": {
            "teacher_correct": sum(teacher[key]["score"] == 1.0 for key in gains),
            "teacher_wrong": sum(teacher[key]["score"] != 1.0 for key in gains),
            "indices": [key[1] for key in gains],
        },
    }
    if source == GSM:
        result["regression_details"]["opd_flexible_correct"] = sum(
            gsm_flexible_answer(opd[key]["response"]) == str(opd[key]["ground_truth"]).replace(",", "")
            for key in regressions
        )
        result["regression_details"]["opd_has_ground_truth_in_tail"] = sum(
            str(opd[key]["ground_truth"]).replace(",", "")
            in opd[key]["response"][-300:].replace(",", "")
            for key in regressions
        )
        result["regression_samples"] = [
            {
                "index": key[1],
                "ground_truth": opd[key]["ground_truth"],
                "finish_reason": opd[key]["finish_reason"],
                "teacher_score": teacher[key]["score"],
                "opd_flexible_answer": gsm_flexible_answer(opd[key]["response"]),
                "opd_response_tail": opd[key]["response"][-400:],
            }
            for key in regressions[:5]
        ]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--opd", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base = load(args.base)
    opd = load(args.opd)
    teacher = load(args.teacher)
    if not (base.keys() == opd.keys() == teacher.keys()):
        raise RuntimeError("Evaluation record keys do not match")

    report = {
        "scores": {
            "qwen3_4b_base": {source: dataset_summary(base, source) for source in (GSM, MATH)},
            "qwen3_4b_opd_step234": {source: dataset_summary(opd, source) for source in (GSM, MATH)},
            "qwen3_14b_base": {source: dataset_summary(teacher, source) for source in (GSM, MATH)},
        },
        "transitions": {
            source: transitions(base, opd, teacher, source) for source in (GSM, MATH)
        },
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
