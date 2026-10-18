#!/usr/bin/env python3
"""Apply GPT-4o-mini k1_t2 scores to the intervened viewer rows."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_js", type=Path)
    parser.add_argument("paired_results", type=Path)
    args = parser.parse_args()
    source = args.data_js.read_text()
    payload = json.loads(source[len("window.CRUX_DATA = "):-2])
    paired = json.loads(args.paired_results.read_text())["paired"]
    scores = {
        (row["dataset"], row["problem_id"]): row
        for row in paired
        if row["config"] == "k1_t2" and row["dataset"] in {"olympiad_bench", "bbh", "math500"}
    }
    changed = 0
    matched = 0
    for item in payload["items"]:
        row = scores.get((item["dataset"], item["id"]))
        if row is None:
            continue
        matched += 1
        result = item["intervened"]
        result["oldCorrect"] = result["correct"]
        result["correct"] = row["new_correct"]
        result["scorer"] = "GPT-4o-mini"
        result["scoreChanged"] = row["old_correct"] != row["new_correct"]
        changed += result["scoreChanged"]
    if matched != len(scores):
        raise SystemExit(f"score coverage mismatch: matched={matched}, expected={len(scores)}")
    payload["meta"]["gpt4oMiniK1T2"] = {
        "rows": matched,
        "changed": changed,
        "datasets": ["bbh", "math500", "olympiad_bench"],
    }
    args.data_js.write_text("window.CRUX_DATA = " + json.dumps(payload, ensure_ascii=False) + ";\n")
    print(f"applied {matched} GPT-4o-mini scores; {changed} labels changed")


if __name__ == "__main__":
    main()
