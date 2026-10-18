#!/usr/bin/env python3
"""Export aligned student / k1_t2 / teacher rollouts for the viewer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CONFIGS = ("student_only", "k1_t2", "teacher_only")


def byte_decoder() -> dict[str, int]:
    values = list(range(ord("!"), ord("~") + 1))
    values += list(range(ord("¡"), ord("¬") + 1))
    values += list(range(ord("®"), ord("ÿ") + 1))
    chars = list(values)
    missing = 0
    for value in range(256):
        if value not in values:
            values.append(value)
            chars.append(256 + missing)
            missing += 1
    return {chr(char): value for char, value in zip(chars, values)}


def load_token_decoder(path: Path):
    spec = json.loads(path.read_text())
    inverse = {int(token_id): token for token, token_id in spec["model"]["vocab"].items()}
    byte_map = byte_decoder()

    def decode(token_id: int) -> str:
        token = inverse.get(token_id, "")
        if token.startswith("<|") and token.endswith("|>"):
            return ""
        raw = bytes(byte_map[char] for char in token if char in byte_map)
        return raw.decode("utf-8", errors="replace")

    return decode


def read_rows(root: Path, config: str) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for path in sorted((root / config / "completed").glob("*.json")):
        row = json.loads(path.read_text())
        if row.get("dataset") and row.get("problem_id"):
            row["_filename"] = path.name
            rows[f"{row['dataset']}::{row['problem_id']}"] = row
    return rows


def read_score(root: Path, config: str, completed_path_name: str) -> bool | None:
    path = root / config / "scores" / completed_path_name
    if not path.exists():
        return None
    return bool(json.loads(path.read_text()).get("correct"))


def compact(row: dict, correct: bool | None, *, include_sources: bool = False, decode=None) -> dict:
    result = {
        "text": row.get("response", ""),
        "tokens": len(row.get("response_token_ids", [])),
        "correct": correct,
        "finish": row.get("finish_reason"),
    }
    if include_sources:
        result["tokenIds"] = row.get("response_token_ids", [])
        result["tokenPieces"] = [decode(value) for value in row.get("response_token_ids", [])]
        result["tokenSources"] = row.get("token_sources", [])
        result["events"] = [
            {
                "round": event.get("round"),
                "anchor": event.get("anchor_response_position"),
                "teacherTokens": event.get("bridge_token_ids", []),
                "threshold": event.get("threshold_candidate", {}),
            }
            for event in row.get("events", [])
            if event.get("bridge_token_ids") or event.get("anchor_response_position") is not None
        ]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="code_reasoning_grid_20260913 root")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("data.js"))
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    tokenizer_path = args.tokenizer
    if tokenizer_path is None:
        tokenizer_path = Path.cwd() / "tmp/anchor_smoke_local/tokenizer/tokenizer.json"
    decode = load_token_decoder(tokenizer_path)

    loaded = {config: read_rows(args.root, config) for config in CONFIGS}
    common = set.intersection(*(set(rows) for rows in loaded.values()))
    if not common:
        raise SystemExit("No aligned rows found")

    items = []
    for key in sorted(common, key=lambda value: (value.split("::", 1)[0], value)):
        base = loaded["student_only"][key]
        name = base["_filename"]
        items.append(
            {
                "id": base["problem_id"],
                "dataset": base["dataset"],
                "datasetIndex": base.get("dataset_index"),
                "problem": base.get("problem", ""),
                "solution": base.get("solution", ""),
                "student": compact(
                    base, read_score(args.root, "student_only", name)
                ),
                "intervened": compact(
                    loaded["k1_t2"][key],
                    read_score(args.root, "k1_t2", name),
                    include_sources=True,
                    decode=decode,
                ),
                "teacher": compact(
                    loaded["teacher_only"][key],
                    read_score(args.root, "teacher_only", name),
                ),
            }
        )

    datasets = sorted({item["dataset"] for item in items})
    summaries = {}
    for dataset in datasets:
        subset = [item for item in items if item["dataset"] == dataset]
        summaries[dataset] = {
            "count": len(subset),
            **{
                config: sum(item[config]["tokens"] for item in subset) / len(subset)
                for config in ("student", "intervened", "teacher")
            },
        }
    payload = {
        "meta": {
            "datasets": datasets,
            "studentModel": "Qwen3-1.7B",
            "teacherModel": "Qwen3-14B",
            "intervention": "k=1, threshold=2",
            "summaries": summaries,
            "count": len(items),
        },
        "items": items,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("window.CRUX_DATA = " + json.dumps(payload, ensure_ascii=False) + ";\n")
    print(f"wrote {len(items)} aligned problems across {len(datasets)} datasets to {args.output}")


if __name__ == "__main__":
    main()
