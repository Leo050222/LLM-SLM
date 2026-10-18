#!/usr/bin/env python3
"""Score final answers without requiring GSM8K #### or MATH boxed syntax.

The primary score is conservative: it accepts an existing boxed/#### answer,
or an answer explicitly emitted after the Qwen thinking block / final-answer
marker. A separate last-number diagnostic is reported for GSM8K only.
"""

import argparse
import json
import math
import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path

from verl.utils.reward_score.math_reward import is_equiv, last_boxed_only_string, remove_boxed


GSM = "gsm8k"
MATH = "math"


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def source(row: dict) -> str:
    return GSM if 'after "####"' in row["input"] else MATH


def boxed_answer(text: str):
    boxed = last_boxed_only_string(text)
    if boxed is None:
        return None
    try:
        return remove_boxed(boxed)
    except Exception:
        return None


def last_number(text: str):
    candidates = re.findall(r"-?(?:\d[\d,]*)(?:\.\d+)?(?:/[0-9,]+)?", text[-600:])
    return candidates[-1] if candidates else None


def explicit_answer_segment(text: str):
    if "</think>" in text:
        segment = text.rsplit("</think>", 1)[1].strip()
        if segment:
            return segment

    tail = text[-800:]
    matches = list(
        re.finditer(
            r"(?is)(?:final\s+answer|answer\s+is|answer\s*:|therefore[, ]+the\s+answer\s+is|thus[, ]+the\s+answer\s+is)\s*[:=]?\s*",
            tail,
        )
    )
    if matches:
        return tail[matches[-1].end() :].strip()
    return None


def clean_math_candidate(segment: str):
    if not segment:
        return None
    boxed = boxed_answer(segment)
    if boxed is not None:
        return boxed

    math_spans = re.findall(r"\$\$([^$]+)\$\$|\$([^$]+)\$", segment, flags=re.S)
    if math_spans:
        candidate = next(value for value in reversed(math_spans[-1]) if value)
    else:
        lines = [line.strip() for line in segment.splitlines() if line.strip()]
        if not lines:
            return None
        candidate = lines[-1]

    candidate = re.sub(r"^[#*\s:=-]+", "", candidate)
    candidate = re.sub(r"[.*\s]+$", "", candidate)
    if "=" in candidate and len(candidate.split("=", 1)[0].strip()) <= 12:
        candidate = candidate.split("=", 1)[1].strip()
    return candidate or None


def extract_primary(row: dict):
    text = row["output"]
    if source(row) == GSM:
        strict = re.findall(r"#### (-?[0-9.,]+)", text[-300:])
        if strict:
            return strict[-1]
        boxed = boxed_answer(text)
        if boxed is not None:
            return boxed
        segment = explicit_answer_segment(text)
        return last_number(segment) if segment else None

    boxed = boxed_answer(text)
    if boxed is not None:
        return boxed
    return clean_math_candidate(explicit_answer_segment(text))


def numeric_value(value):
    if value is None:
        return None
    text = str(value).strip().replace(",", "").replace("$", "")
    text = text.replace("\\%", "%").replace("%", "")
    text = re.sub(r"\\text\{[^{}]*\}", "", text).strip()
    frac = re.fullmatch(r"\\(?:d?frac|tfrac)\{(-?\d+)\}\{(\d+)\}", text)
    try:
        if frac:
            return Fraction(int(frac.group(1)), int(frac.group(2)))
        if re.fullmatch(r"-?\d+/\d+", text):
            return Fraction(text)
        return Fraction(Decimal(text))
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None


def format_free_score(row: dict) -> bool:
    candidate = extract_primary(row)
    if source(row) == GSM:
        return numeric_value(candidate) == numeric_value(row["gts"]) and candidate is not None
    return candidate is not None and is_equiv(candidate, row["gts"])


def last_number_score(row: dict) -> bool:
    candidate = last_number(row["output"])
    return numeric_value(candidate) == numeric_value(row["gts"]) and candidate is not None


def exact_mcnemar_p(regressions: int, gains: int) -> float:
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
    parser.add_argument("--records-output", type=Path, required=True)
    args = parser.parse_args()

    models = {"qwen3_4b_base": load(args.base), "qwen3_4b_opd": load(args.opd), "qwen3_14b_base": load(args.teacher)}
    lengths = {len(rows) for rows in models.values()}
    if lengths != {512}:
        raise RuntimeError(f"Expected three aligned 512-row dumps, got {lengths}")
    for i in range(512):
        if len({rows[i]["input"] for rows in models.values()}) != 1:
            raise RuntimeError(f"Unaligned prompt at row {i}")

    scored: dict[str, list[dict]] = {}
    report = {"definition": "final declared answer; no required ####/boxed wrapper", "scores": {}, "paired_opd_vs_base": {}}
    for name, rows in models.items():
        scored[name] = []
        report["scores"][name] = {}
        for i, row in enumerate(rows):
            record = {
                "row": i,
                "source": source(row),
                "ground_truth": row["gts"],
                "strict_correct": row["score"] == 1.0,
                "format_free_candidate": extract_primary(row),
                "format_free_correct": format_free_score(row),
            }
            if source(row) == GSM:
                record["last_number_correct"] = last_number_score(row)
            scored[name].append(record)

        for dataset in (GSM, MATH):
            subset = [row for row in scored[name] if row["source"] == dataset]
            item = {
                "total": len(subset),
                "strict_correct": sum(row["strict_correct"] for row in subset),
                "format_free_correct": sum(row["format_free_correct"] for row in subset),
                "format_free_accuracy": sum(row["format_free_correct"] for row in subset) / len(subset),
                "recovered_by_format_free": sum(row["format_free_correct"] and not row["strict_correct"] for row in subset),
                "candidate_missing": sum(row["format_free_candidate"] is None for row in subset),
            }
            if dataset == GSM:
                item["last_number_correct"] = sum(row["last_number_correct"] for row in subset)
                item["last_number_accuracy"] = item["last_number_correct"] / len(subset)
            report["scores"][name][dataset] = item

    for dataset in (GSM, MATH):
        indices = [i for i, row in enumerate(scored["qwen3_4b_base"]) if row["source"] == dataset]
        regressions = sum(
            scored["qwen3_4b_base"][i]["format_free_correct"] and not scored["qwen3_4b_opd"][i]["format_free_correct"]
            for i in indices
        )
        gains = sum(
            not scored["qwen3_4b_base"][i]["format_free_correct"] and scored["qwen3_4b_opd"][i]["format_free_correct"]
            for i in indices
        )
        report["paired_opd_vs_base"][dataset] = {
            "regressions": regressions,
            "gains": gains,
            "net": gains - regressions,
            "mcnemar_exact_p": exact_mcnemar_p(regressions, gains),
        }

    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with args.records_output.open("w", encoding="utf-8") as handle:
        for name, records in scored.items():
            for record in records:
                handle.write(json.dumps({"model": name, **record}, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
