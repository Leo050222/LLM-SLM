#!/usr/bin/env python3
"""Apply per-problem GPT-4o-mini student/teacher baseline scores to data.js."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_js", type=Path)
    parser.add_argument("scores_json", type=Path)
    args = parser.parse_args()
    source = args.data_js.read_text()
    payload = json.loads(source[len("window.CRUX_DATA = "):-2])
    scores = {
        (row["dataset"], row["config"], row["problem_id"]): row
        for row in json.loads(args.scores_json.read_text())["rows"]
        if row["dataset"] in {"bbh", "math500", "olympiad_bench"}
    }
    matched = changed = 0
    for item in payload["items"]:
        for config, field in (("student_only", "student"), ("teacher_only", "teacher")):
            row = scores.get((item["dataset"], config, item["id"]))
            if row is None:
                continue
            matched += 1
            result = item[field]
            result["oldCorrect"] = result["correct"]
            result["correct"] = row["new_correct"]
            result["scorer"] = "GPT-4o-mini"
            result["scoreChanged"] = row["old_correct"] != row["new_correct"]
            changed += result["scoreChanged"]
    expected = len(scores)
    if matched != expected:
        raise SystemExit(f"score coverage mismatch: matched={matched}, expected={expected}")
    payload["meta"]["gpt4oMiniBaselines"] = {"rows": matched, "changed": changed}
    args.data_js.write_text("window.CRUX_DATA = " + json.dumps(payload, ensure_ascii=False) + ";\n")
    print(f"applied {matched} baseline scores; {changed} labels changed")


if __name__ == "__main__":
    main()
