#!/usr/bin/env python3
"""Print representative MATH responses that lack a boxed final answer."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--closed-think", action="store_true")
    args = parser.parse_args()

    shown = 0
    for line in args.path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if "within \\boxed{}" not in row["input"] or "\\boxed" in row["output"]:
            continue
        if args.closed_think and "</think>" not in row["output"]:
            continue
        print(json.dumps({"gts": row["gts"], "score": row["score"], "tail": row["output"][-650:]}, ensure_ascii=False))
        shown += 1
        if shown >= args.limit:
            break


if __name__ == "__main__":
    main()
