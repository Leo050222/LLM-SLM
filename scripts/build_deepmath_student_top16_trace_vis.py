#!/usr/bin/env python3
"""Build a self-contained token heatmap for Student-Top16 trace diagnostics."""

from __future__ import annotations

import argparse
import base64
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--diagnostic-root", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tokens = pd.read_parquet(args.diagnostic_root / "tokens.parquet")
    student_rows = read_jsonl(args.run_root / "merged_qwen3_1_7b" / "qwen3_1_7b_rollouts.jsonl")
    teacher_rows = read_jsonl(args.run_root / "merged" / "qwen3_14b_rollouts.jsonl")
    student = {int(row["dataset_index"]): row for row in student_rows}
    teacher = {int(row["dataset_index"]): row for row in teacher_rows}
    indices = sorted(int(value) for value in tokens["dataset_index"].unique())
    if len(indices) != 209:
        raise RuntimeError(f"Expected 209 selected rows, found {len(indices)}")

    metrics = {
        "top16kl": "student_top16_renorm_kl",
        "overlap": "student_teacher_top16_overlap",
        "k1": "sample_logprob_gap_student_minus_teacher",
        "k3": "sampled_reverse_kl_k3",
    }
    caps = {}
    for name, column in metrics.items():
        values = tokens[column].to_numpy(float)
        values = np.abs(values) if name == "k1" else np.maximum(0, values)
        caps[name] = round(float(max(1e-8, np.quantile(values, 0.995))), 8)

    cases = []
    streams = {}
    for index in indices:
        group = tokens[tokens["dataset_index"] == index].sort_values("response_position")
        row = student[index]
        cases.append({
            "index": index,
            "problemId": str(row["problem_id"]),
            "difficulty": float(row["difficulty"]),
            "topic": str(row["topic"]),
            "studentTokens": int(row["response_tokens"]),
            "teacherTokens": int(teacher[index]["response_tokens"]),
            "studentAnswer": str(row.get("predicted_answer") or "未抽取"),
            "teacherAnswer": str(teacher[index].get("predicted_answer") or "未抽取"),
            **{name: round(float(group[column].mean()), 6) for name, column in metrics.items()},
        })
        streams[str(index)] = {
            "tokens": group["token_text"].fillna("").astype(str).tolist(),
            **{
                name: np.round(group[column].to_numpy(float), 6).tolist()
                for name, column in metrics.items()
            },
            "studentLogp": np.round(group["student_sample_logprob"].to_numpy(float), 6).tolist(),
            "teacherLogp": np.round(group["teacher_sample_logprob"].to_numpy(float), 6).tolist(),
            "top1": group["top1_agreement"].to_numpy(np.uint8).tolist(),
        }
    payload = {
        "protocol": {
            "selected": 209,
            "total": 1000,
            "supportTopK": 16,
            "support": "student-selected, renormalized",
            "condition": "Qwen3-1.7B wrong and Qwen3-14B correct",
        },
        "caps": caps,
        "cases": cases,
        "streams": streams,
        "windowSize": 512,
    }
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    encoded = base64.b64encode(gzip.compress(raw, compresslevel=9)).decode("ascii")
    template = args.template.read_text(encoding="utf-8")
    if template.count("__PAYLOAD_B64__") != 1:
        raise RuntimeError("Template must contain exactly one payload marker")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(template.replace("__PAYLOAD_B64__", encoded), encoding="utf-8")
    print(json.dumps({
        "status": "STUDENT_TOP16_TRACE_VIS_OK",
        "cases": len(cases),
        "tokens": int(len(tokens)),
        "html_bytes": args.output.stat().st_size,
    }))


if __name__ == "__main__":
    main()
