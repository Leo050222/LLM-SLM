#!/usr/bin/env python3
"""Build the focused token-logit viewer for 1.7B-wrong / 14B-correct cases."""

from __future__ import annotations

import argparse
import base64
import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXPECTED_SELECTED = 209
METRICS = {
    "k1": "sampled_reverse_kl_k1",
    "k3": "sampled_reverse_kl_k3",
    "fwd": "raw_forward_kl_top128",
    "top1": "top1_agreement",
    "overlap": "topk_overlap",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def rounded(value: float) -> float:
    return round(float(value), 6)


def aggregate(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "n": int(len(frame)),
        **{name: rounded(frame[column].mean()) for name, column in METRICS.items()},
    }


def token_path(root: Path, dataset_index: int) -> Path:
    return (
        root
        / "diagnostic_qwen3_1_7b"
        / f"shard_{dataset_index % 4}"
        / "token_shards"
        / f"example_{dataset_index:03d}.parquet"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.root
    student_rows = read_jsonl(
        root / "merged_qwen3_1_7b" / "qwen3_1_7b_rollouts.jsonl"
    )
    teacher_rows = read_jsonl(root / "merged" / "qwen3_14b_rollouts.jsonl")
    student = {int(row["dataset_index"]): row for row in student_rows}
    teacher = {int(row["dataset_index"]): row for row in teacher_rows}
    if set(student) != set(teacher) or len(student) != 1000:
        raise RuntimeError("Student and teacher rollouts are not aligned 1,000-row sets")

    selected_indices = sorted(
        index
        for index in student
        if not student[index]["correct"] and teacher[index]["correct"]
    )
    if len(selected_indices) != EXPECTED_SELECTED:
        raise RuntimeError(
            f"Expected {EXPECTED_SELECTED} selected rows, found {len(selected_indices)}"
        )

    diagnostics = pd.read_parquet(
        root / "merged_qwen3_1_7b" / "per_rollout_diagnostics.parquet"
    ).set_index("dataset_index")
    selected_diag = diagnostics.loc[selected_indices].copy()

    overview = {
        "all": aggregate(diagnostics),
        "selected": aggregate(selected_diag),
    }

    difficulty: list[dict[str, Any]] = []
    for value in sorted(float(item) for item in diagnostics["difficulty"].unique()):
        all_group = diagnostics[diagnostics["difficulty"] == value]
        group = selected_diag[selected_diag["difficulty"] == value]
        difficulty.append(
            {
                "difficulty": value,
                "allN": int(len(all_group)),
                "selectedN": int(len(group)),
                "selectedRate": rounded(len(group) / len(all_group)),
                **(
                    {name: rounded(group[column].mean()) for name, column in METRICS.items()}
                    if len(group)
                    else {name: None for name in METRICS}
                ),
            }
        )

    topic_all = diagnostics.groupby("topic", observed=True).size()
    topic_selected = selected_diag.groupby("topic", observed=True)
    topics: list[dict[str, Any]] = []
    for topic, group in topic_selected:
        topics.append(
            {
                "topic": str(topic),
                "n": int(len(group)),
                "allN": int(topic_all.loc[topic]),
                "rate": rounded(len(group) / int(topic_all.loc[topic])),
                "k1": rounded(group[METRICS["k1"]].mean()),
                "k3": rounded(group[METRICS["k3"]].mean()),
                "top1": rounded(group[METRICS["top1"]].mean()),
                "overlap": rounded(group[METRICS["overlap"]].mean()),
            }
        )
    topics.sort(key=lambda row: (-row["n"], -row["k1"], row["topic"]))
    topics = topics[:10]

    case_candidates = [
        (int(selected_diag[METRICS["k1"]].idxmax()), "rollout K1 最高"),
        (int(selected_diag[METRICS["k3"]].idxmax()), "rollout K3 最高"),
        (int(selected_diag[METRICS["top1"]].idxmin()), "Top-1 一致率最低"),
        (int(selected_diag[METRICS["overlap"]].idxmin()), "Top-128 overlap 最低"),
        (int(selected_diag["response_tokens"].idxmax()), "1.7B 输出最长"),
        (int(selected_diag[METRICS["k1"]].idxmin()), "做错但 logits 最接近"),
    ]
    seen: set[int] = set()
    cases: list[tuple[int, str]] = []
    for index, reason in case_candidates:
        if index not in seen:
            cases.append((index, reason))
            seen.add(index)
    for index in selected_diag[METRICS["k1"]].sort_values(ascending=False).index:
        if len(cases) >= 6:
            break
        if int(index) not in seen:
            cases.append((int(index), "高 K1 补充案例"))
            seen.add(int(index))

    needed_columns = [
        "dataset_index",
        "response_position",
        "relative_position",
        "region",
        "token_text",
        "student_chosen_logprob",
        "sampled_reverse_kl_k1",
        "sampled_reverse_kl_k3",
        "raw_forward_kl_top128",
        "top1_agreement",
        "topk_overlap",
    ]
    token_frames: list[pd.DataFrame] = []
    for index in selected_indices:
        path = token_path(root, index)
        if not path.is_file():
            raise FileNotFoundError(path)
        token_frames.append(pd.read_parquet(path, columns=needed_columns))
    tokens = pd.concat(token_frames, ignore_index=True)
    numeric = tokens.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy()).all():
        raise RuntimeError("Selected token diagnostics contain NaN or infinity")

    tokens["quartile"] = np.minimum(
        (tokens["relative_position"] * 4).astype(int), 3
    )
    position = [
        {"quartile": int(quartile), "tokens": int(len(group)), **aggregate(group)}
        for quartile, group in tokens.groupby("quartile", observed=True)
    ]
    regions = [
        {"region": str(region), "tokens": int(len(group)), **aggregate(group)}
        for region, group in tokens.groupby("region", observed=True)
    ]

    case_rows: list[dict[str, Any]] = []
    streams: dict[str, Any] = {}
    for index, reason in cases:
        row = student[index]
        diag = selected_diag.loc[index]
        case_rows.append(
            {
                "index": index,
                "problemId": str(row["problem_id"]),
                "difficulty": float(row["difficulty"]),
                "topic": str(row["topic"]),
                "reason": reason,
                "studentTokens": int(row["response_tokens"]),
                "teacherTokens": int(teacher[index]["response_tokens"]),
                "studentAnswer": str(row.get("predicted_answer") or "未抽取"),
                "teacherAnswer": str(teacher[index].get("predicted_answer") or "未抽取"),
                "k1": rounded(diag[METRICS["k1"]]),
                "k3": rounded(diag[METRICS["k3"]]),
                "top1": rounded(diag[METRICS["top1"]]),
                "overlap": rounded(diag[METRICS["overlap"]]),
            }
        )
        group = tokens[tokens["dataset_index"] == index].sort_values(
            "response_position"
        )
        streams[str(index)] = {
            "tokens": group["token_text"].fillna("").astype(str).tolist(),
            "k1": np.round(group[METRICS["k1"]].to_numpy(float), 5).tolist(),
            "k3": np.round(group[METRICS["k3"]].to_numpy(float), 5).tolist(),
            "fwd": np.round(group[METRICS["fwd"]].to_numpy(float), 5).tolist(),
            "studentLogp": np.round(
                group["student_chosen_logprob"].to_numpy(float), 5
            ).tolist(),
            "top1": group[METRICS["top1"]].to_numpy(np.uint8).tolist(),
            "overlap": np.round(
                group[METRICS["overlap"]].to_numpy(float), 4
            ).tolist(),
        }

    payload = {
        "protocol": {
            "selected": EXPECTED_SELECTED,
            "total": 1000,
            "condition": "Qwen3-1.7B wrong and Qwen3-14B correct",
            "maxResponseTokens": 40512,
            "maxModelLen": 40960,
            "temperature": 1.0,
            "topP": 1.0,
            "thinking": True,
        },
        "overview": overview,
        "difficulty": difficulty,
        "position": position,
        "regions": regions,
        "topics": topics,
        "cases": case_rows,
        "streams": streams,
        "windowSize": 512,
    }
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    encoded = base64.b64encode(gzip.compress(raw, compresslevel=9)).decode("ascii")
    template = args.template.read_text(encoding="utf-8")
    if template.count("__PAYLOAD_B64__") != 1:
        raise RuntimeError("Template must contain exactly one __PAYLOAD_B64__")
    output_text = template.replace("__PAYLOAD_B64__", encoded)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output_text, encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "DEEPMATH_17B_MISS14B_VIS_OK",
                "selected": len(selected_indices),
                "selected_tokens": int(len(tokens)),
                "cases": len(case_rows),
                "raw_payload_bytes": len(raw),
                "html_bytes": args.output.stat().st_size,
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
