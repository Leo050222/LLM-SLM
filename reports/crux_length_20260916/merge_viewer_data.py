#!/usr/bin/env python3
"""Merge an exported supplemental JSON file into data.js."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_js", type=Path)
    parser.add_argument("extra_json", type=Path)
    args = parser.parse_args()
    source = args.data_js.read_text()
    payload = json.loads(source[len("window.CRUX_DATA = "):-2])
    extra = json.loads(args.extra_json.read_text())["items"]
    existing = {(item["dataset"], item["id"]) for item in payload["items"]}
    payload["items"].extend(item for item in extra if (item["dataset"], item["id"]) not in existing)
    payload["items"].sort(key=lambda item: (item["dataset"], item["id"]))
    datasets = sorted({item["dataset"] for item in payload["items"]})
    payload["meta"]["datasets"] = datasets
    payload["meta"]["count"] = len(payload["items"])
    payload["meta"]["summaries"] = {
        dataset: {
            "count": len(rows := [item for item in payload["items"] if item["dataset"] == dataset]),
            **{key: sum(item[key]["tokens"] for item in rows) / len(rows) for key in ("student", "intervened", "teacher")},
        }
        for dataset in datasets
    }
    args.data_js.write_text("window.CRUX_DATA = " + json.dumps(payload, ensure_ascii=False) + ";\n")
    print(f"merged {len(extra)} rows; total {len(payload['items'])}")


if __name__ == "__main__":
    main()
