#!/usr/bin/env python3
"""CPU-only preflight for every selected anchor before bridge generation."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR.parent) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR.parent))

from bridge_generator.run_bridge_batch import (  # noqa: E402
    DEFAULT_TEACHER,
    discover_selected_anchors,
    prepare_record,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-root", type=Path, required=True)
    parser.add_argument("--teacher", default=DEFAULT_TEACHER)
    parser.add_argument("--tokenizer")
    parser.add_argument("--h", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--expected-count", type=int)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    rows = discover_selected_anchors(args.anchor_root)
    if args.expected_count is not None and len(rows) != args.expected_count:
        raise ValueError(
            f"selected anchor count {len(rows)} != expected {args.expected_count}"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer or args.teacher,
        local_files_only=True,
        trust_remote_code=False,
    )
    prepared = [
        prepare_record(path, anchor, tokenizer, args.h, args.max_model_len)
        for path, anchor in rows
    ]
    lengths = [len(row["prefix_ids"]) for row in prepared]
    indices = [int(row["anchor"]["dataset_index"]) for row in prepared]
    if len(indices) != len(set(indices)):
        raise ValueError("duplicate dataset_index among selected anchors")
    shard_counts = [sum(index % 8 == shard for index in indices) for shard in range(8)]
    summary = {
        "status": "BRIDGE_INPUT_AUDIT_PASSED",
        "selected_anchor_count": len(rows),
        "requested_bridge_length": args.h,
        "max_model_len": args.max_model_len,
        "prefix_length_min": min(lengths),
        "prefix_length_median": statistics.median(lengths),
        "prefix_length_max": max(lengths),
        "single_node_eight_shard_counts": shard_counts,
    }
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
