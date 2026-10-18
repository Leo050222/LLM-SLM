#!/usr/bin/env python3
"""Export aligned k1_t2 math/reasoning rows using historical baseline roots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_viewer_data import compact, load_token_decoder


def load_group(
    completed: Path,
    scores: Path,
    datasets: set[str],
    default_dataset: str | None = None,
) -> dict[str, tuple[dict, bool | None]]:
    rows = {}
    for path in completed.glob("*.json"):
        row = json.loads(path.read_text())
        dataset = row.get("dataset") or default_dataset
        problem_id = row.get("problem_id")
        if dataset and problem_id and default_dataset and not problem_id.startswith(f"{dataset}/"):
            row = dict(row)
            row["dataset"] = dataset
            row["problem_id"] = f"{dataset}/{problem_id}"
            problem_id = row["problem_id"]
        if dataset not in datasets or not problem_id:
            continue
        score_path = scores / path.name
        correct = None
        if score_path.exists():
            correct = bool(json.loads(score_path.read_text()).get("correct"))
        rows[f"{dataset}::{problem_id}"] = (row, correct)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("hard_root", type=Path)
    parser.add_argument("grid_root", type=Path)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["olympiad_bench", "bbh", "math500"],
        help="Datasets to export (default: olympiad_bench bbh math500)",
    )
    args = parser.parse_args()
    decode = load_token_decoder(args.tokenizer)
    datasets = set(args.datasets)
    default_dataset = next(iter(datasets)) if len(datasets) == 1 else None

    groups = {
        "student": load_group(args.hard_root / "student/completed", args.hard_root / "scores/student", datasets, default_dataset),
        "teacher": load_group(args.hard_root / "teacher/completed", args.hard_root / "scores/teacher", datasets, default_dataset),
        "intervened": load_group(args.grid_root / "k1_t2/completed", args.grid_root / "scores/k1_t2", datasets, default_dataset),
    }
    common = set.intersection(*(set(group) for group in groups.values()))
    items = []
    for key in sorted(common):
        base, student_correct = groups["student"][key]
        intervened, intervened_correct = groups["intervened"][key]
        teacher, teacher_correct = groups["teacher"][key]
        items.append({
            "id": base["problem_id"],
            "dataset": base["dataset"],
            "datasetIndex": base.get("dataset_index"),
            "problem": base.get("problem", ""),
            "solution": base.get("solution", ""),
            "student": compact(base, student_correct),
            "intervened": compact(intervened, intervened_correct, include_sources=True, decode=decode),
            "teacher": compact(teacher, teacher_correct),
        })
    args.output.write_text(json.dumps({"items": items}, ensure_ascii=False))
    counts = {dataset: sum(item["dataset"] == dataset for item in items) for dataset in sorted(datasets)}
    print(f"wrote {len(items)} rows: {counts}")


if __name__ == "__main__":
    main()
