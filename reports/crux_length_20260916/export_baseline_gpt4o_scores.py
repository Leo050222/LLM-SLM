#!/usr/bin/env python3
"""Export per-problem GPT-4o-mini baseline scores from the remote rescore root."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=None)
    args = parser.parse_args()
    rows = []
    for line in (args.root / "inputs.jsonl").read_text().splitlines():
        item = json.loads(line)
        if args.datasets and item["dataset"] not in set(args.datasets):
            continue
        path = args.root / "scores" / item["dataset"] / item["config"] / (Path(item["relative_path"]).stem + ".json")
        score = json.loads(path.read_text())
        assert score["source_sha256"] == item["source_sha256"]
        assert score["input_sha256"] == item["input_sha256"]
        rows.append({
            "dataset": item["dataset"],
            "config": item["config"],
            "problem_id": item["problem_id"],
            "old_correct": item["old_correct"],
            "new_correct": score["is_correct"],
        })
    args.output.write_text(json.dumps({"rows": rows}, ensure_ascii=False))
    print(f"exported {len(rows)} baseline scores")


if __name__ == "__main__":
    main()
