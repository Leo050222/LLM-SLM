#!/usr/bin/env python3
"""Build a compact DeepMath token-difference payload for the OPD viewer."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


MODEL_PATHS = {
    "1.7B": {
        "rollouts": "merged_qwen3_1_7b/qwen3_1_7b_rollouts.jsonl",
        "per_rollout": "merged_qwen3_1_7b/per_rollout_diagnostics.parquet",
        "tokens": "merged_qwen3_1_7b/tokens.parquet",
        "summary": "merged_qwen3_1_7b/summary.json",
        "summary_key": "qwen3_1_7b",
    },
    "4B": {
        "rollouts": "merged/qwen3_4b_rollouts.jsonl",
        "per_rollout": "merged/per_rollout_diagnostics.parquet",
        "tokens": "merged/tokens.parquet",
        "summary": "merged/summary.json",
        "summary_key": "qwen3_4b",
    },
}

PATTERN_LABELS = {
    "000": "三个模型都错",
    "001": "仅 14B 正确",
    "010": "仅 4B 正确",
    "011": "4B 与 14B 正确",
    "100": "仅 1.7B 正确",
    "101": "1.7B 与 14B 正确",
    "110": "1.7B 与 4B 正确",
    "111": "三个模型都正确",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def rounded(value: float) -> float:
    return round(float(value), 6)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root

    rollouts = {
        model: read_jsonl(root / spec["rollouts"])
        for model, spec in MODEL_PATHS.items()
    }
    teacher_rows = read_jsonl(root / "merged/qwen3_14b_rollouts.jsonl")
    rollout_maps = {
        model: {int(row["dataset_index"]): row for row in rows}
        for model, rows in rollouts.items()
    }
    teacher_map = {int(row["dataset_index"]): row for row in teacher_rows}

    diagnostics = {
        model: pd.read_parquet(root / spec["per_rollout"]).set_index("dataset_index")
        for model, spec in MODEL_PATHS.items()
    }
    joined = diagnostics["1.7B"].add_suffix("_17").join(
        diagnostics["4B"].add_suffix("_4"), how="inner"
    )
    joined["combined_k1"] = (
        joined["sampled_reverse_kl_k1_17"]
        + joined["sampled_reverse_kl_k1_4"]
    )

    def pattern(index: int) -> str:
        return "".join(
            "1" if value else "0"
            for value in (
                rollout_maps["1.7B"][index]["correct"],
                rollout_maps["4B"][index]["correct"],
                teacher_map[index]["correct"],
            )
        )

    joined["pattern"] = [pattern(int(index)) for index in joined.index]
    selected: list[tuple[int, str]] = []
    for value in PATTERN_LABELS:
        group = joined[joined["pattern"] == value]
        if not group.empty:
            selected.append((int(group["combined_k1"].idxmax()), value))
    all_correct = joined[joined["pattern"] == "111"]
    if not all_correct.empty:
        selected.append((int(all_correct["combined_k1"].idxmin()), "111-low"))
    hardest = joined[joined["difficulty_17"] == joined["difficulty_17"].max()]
    if not hardest.empty:
        selected.append((int(hardest["combined_k1"].idxmax()), "difficulty-max"))

    seen: set[int] = set()
    selected = [item for item in selected if not (item[0] in seen or seen.add(item[0]))]
    selected_indices = [index for index, _ in selected]

    summaries = {
        model: json.loads((root / spec["summary"]).read_text(encoding="utf-8"))
        for model, spec in MODEL_PATHS.items()
    }
    models: dict[str, Any] = {}
    for model, spec in MODEL_PATHS.items():
        summary = summaries[model]
        native = summary[spec["summary_key"]]
        token_diag = summary["token_diagnostics"]
        models[model] = {
            "correct": native["correct"],
            "accuracy": native["accuracy"],
            "meanTokens": native["response_tokens"]["mean"],
            "length": native["finish_reason"].get("length", 0),
            "k1": token_diag["rollout_weighted"]["sampled_reverse_kl_k1"]["mean"],
            "k3": token_diag["rollout_weighted"]["sampled_reverse_kl_k3"]["mean"],
            "top1": token_diag["rollout_weighted"]["top1_agreement"]["mean"],
            "overlap": token_diag["rollout_weighted"]["topk_overlap"]["mean"],
        }

    difficulty_values = sorted(
        set(float(value) for value in diagnostics["1.7B"]["difficulty"])
    )
    difficulty: list[dict[str, Any]] = []
    for value in difficulty_values:
        row: dict[str, Any] = {"difficulty": value}
        for model in MODEL_PATHS:
            group = diagnostics[model][diagnostics[model]["difficulty"] == value]
            native = [
                rollout_maps[model][int(index)]["correct"] for index in group.index
            ]
            row[model] = {
                "n": int(len(group)),
                "accuracy": rounded(np.mean(native)),
                "k1": rounded(group["sampled_reverse_kl_k1"].mean()),
                "top1": rounded(group["top1_agreement"].mean()),
            }
        difficulty.append(row)

    correctness: dict[str, Any] = {}
    for model, frame in diagnostics.items():
        correctness[model] = {
            str(bool(value)).lower(): {
                "n": int(len(group)),
                "k1": rounded(group["sampled_reverse_kl_k1"].mean()),
                "k3": rounded(group["sampled_reverse_kl_k3"].mean()),
                "top1": rounded(group["top1_agreement"].mean()),
            }
            for value, group in frame.groupby("correct", observed=True)
        }

    topic_frames: dict[str, pd.DataFrame] = {}
    for model, frame in diagnostics.items():
        topic_frames[model] = frame.groupby("topic", observed=True).agg(
            n=("correct", "size"),
            accuracy=("correct", "mean"),
            k1=("sampled_reverse_kl_k1", "mean"),
            top1=("top1_agreement", "mean"),
        )
    topic_joined = topic_frames["1.7B"].add_suffix("_17").join(
        topic_frames["4B"].add_suffix("_4")
    )
    topic_joined = topic_joined[topic_joined["n_17"] >= 10].copy()
    topic_joined["gap"] = topic_joined["k1_17"] - topic_joined["k1_4"]
    topics = [
        {
            "topic": str(topic),
            "n": int(row["n_17"]),
            "k1_17": rounded(row["k1_17"]),
            "k1_4": rounded(row["k1_4"]),
            "gap": rounded(row["gap"]),
            "accuracy_17": rounded(row["accuracy_17"]),
            "accuracy_4": rounded(row["accuracy_4"]),
        }
        for topic, row in topic_joined.sort_values("gap", ascending=False)
        .head(8)
        .iterrows()
    ]

    position: dict[str, list[dict[str, Any]]] = {}
    streams: dict[str, Any] = {}
    for model, spec in MODEL_PATHS.items():
        token_path = root / spec["tokens"]
        position_frame = pd.read_parquet(
            token_path,
            columns=[
                "relative_position",
                "sampled_reverse_kl_k1",
                "sampled_reverse_kl_k3",
                "top1_agreement",
                "topk_overlap",
            ],
        )
        position_frame["quartile"] = np.minimum(
            (position_frame["relative_position"] * 4).astype(int), 3
        )
        position[model] = [
            {
                "quartile": int(quartile),
                "k1": rounded(group["sampled_reverse_kl_k1"].mean()),
                "k3": rounded(group["sampled_reverse_kl_k3"].mean()),
                "top1": rounded(group["top1_agreement"].mean()),
                "overlap": rounded(group["topk_overlap"].mean()),
            }
            for quartile, group in position_frame.groupby("quartile", observed=True)
        ]
        del position_frame

        selected_tokens = pd.read_parquet(
            token_path,
            columns=[
                "dataset_index",
                "response_position",
                "token_text",
                "raw_forward_kl_top128",
                "sampled_reverse_kl_k1",
                "sampled_reverse_kl_k3",
            ],
            filters=[("dataset_index", "in", selected_indices)],
        ).sort_values(["dataset_index", "response_position"])
        for dataset_index, group in selected_tokens.groupby(
            "dataset_index", observed=True, sort=True
        ):
            streams[f"{model}:{int(dataset_index)}"] = {
                "tokens": group["token_text"].fillna("").astype(str).tolist(),
                "fwd": np.round(
                    group["raw_forward_kl_top128"].to_numpy(dtype=float), 5
                ).tolist(),
                "k1": np.round(
                    group["sampled_reverse_kl_k1"].to_numpy(dtype=float), 5
                ).tolist(),
                "k3": np.round(
                    group["sampled_reverse_kl_k3"].to_numpy(dtype=float), 5
                ).tolist(),
            }

    cases: list[dict[str, Any]] = []
    for index, selected_reason in selected:
        base = rollout_maps["4B"][index]
        value = pattern(index)
        reason = PATTERN_LABELS.get(value, value)
        if selected_reason == "111-low":
            reason = "三个模型都正确 · 低差异"
        elif selected_reason == "difficulty-max":
            reason = "最高难度 · 高差异"
        cases.append(
            {
                "index": index,
                "problem": str(base["problem_id"]),
                "difficulty": float(base["difficulty"]),
                "topic": str(base["topic"]),
                "pattern": value,
                "reason": reason,
                "correct": {
                    "1.7B": bool(rollout_maps["1.7B"][index]["correct"]),
                    "4B": bool(rollout_maps["4B"][index]["correct"]),
                    "14B": bool(teacher_map[index]["correct"]),
                },
            }
        )

    payload = {
        "models": models,
        "difficulty": difficulty,
        "position": position,
        "correctness": correctness,
        "topics": topics,
        "cases": cases,
        "streams": streams,
        "windowSize": 512,
        "protocol": {
            "sample": 1000,
            "temperature": 1.0,
            "topP": 1.0,
            "maxTokens": 4096,
            "thinking": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    with gzip.open(args.output, "wb", compresslevel=9) as handle:
        handle.write(encoded)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "raw_bytes": len(encoded),
                "gzip_bytes": args.output.stat().st_size,
                "cases": len(cases),
                "streams": len(streams),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
