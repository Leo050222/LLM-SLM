#!/usr/bin/env python3
"""Merge sampled-token log-probabilities and build a lazy local visualization."""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoTokenizer


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--student-label", default="Qwen3-1.7B baseline")
    parser.add_argument("--teacher-label", default="Qwen3-14B frozen")
    parser.add_argument("--page-title", default="DeepMath Baseline · Sampled-token Log-probability Gap")
    parser.add_argument("--page-heading", default="DeepMath baseline · sampled-token log-probability gap")
    args = parser.parse_args()

    rows = read_jsonl(args.run_root / "merged" / "rollouts.jsonl")
    rows.sort(key=lambda row: int(row["dataset_index"]))
    if len(rows) != 1000 or [int(row["dataset_index"]) for row in rows] != list(range(1000)):
        raise RuntimeError("Baseline rollouts must cover exactly dataset indices 0..999")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = args.output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    cases: list[dict[str, Any]] = []
    all_gaps: list[np.ndarray] = []
    total_bytes = 0
    for row in rows:
        index = int(row["dataset_index"])
        shard = index % 4
        base = args.score_root / f"shard_{shard}"
        with np.load(base / "student_logprobs" / f"example_{index:04d}.npz") as value:
            student = value["sample_logprob"].astype(np.float32)
        with np.load(base / "teacher_logprobs" / f"example_{index:04d}.npz") as value:
            teacher = value["sample_logprob"].astype(np.float32)
        response_ids = np.asarray(row["response_token_ids"], dtype=np.int32)
        if student.shape != teacher.shape or student.shape != response_ids.shape:
            raise RuntimeError(f"index={index}: score/token shape mismatch")
        gap = student - teacher
        if not np.isfinite(gap).all():
            raise RuntimeError(f"index={index}: non-finite gap")
        all_gaps.append(gap)
        token_text = [tokenizer.decode([int(token_id)], skip_special_tokens=False) for token_id in response_ids]
        scale = max(float(np.abs(gap).max()), 1e-8)
        quantized = np.clip(np.rint(gap / scale * 32767.0), -32767, 32767).astype("<i2")
        raw = bytearray(struct.pack("<If", len(gap), scale))
        for value, text in zip(quantized, token_text, strict=True):
            encoded_text = text.encode("utf-8")
            if len(encoded_text) > 65535:
                raise RuntimeError(f"index={index}: single token text is too large")
            raw.extend(struct.pack("<hH", int(value), len(encoded_text)))
            raw.extend(encoded_text)
        encoded = base64.b64encode(gzip.compress(raw, compresslevel=9)).decode("ascii")
        target = data_dir / f"case_{index:04d}.js"
        target.write_text(f'window.__dmGapReceive("{index}","{encoded}");\n', encoding="utf-8")
        total_bytes += target.stat().st_size
        cases.append({
            "index": index,
            "problemId": str(row["problem_id"]),
            "problem": str(row["problem"]),
            "difficulty": float(row["difficulty"]),
            "topic": str(row["topic"]),
            "correct": bool(row["correct"]),
            "tokens": int(len(gap)),
            "meanGap": round(float(gap.mean()), 6),
            "meanAbsGap": round(float(np.abs(gap).mean()), 6),
            "positiveFraction": round(float((gap > 0).mean()), 6),
            "p95AbsGap": round(float(np.quantile(np.abs(gap), 0.95)), 6),
        })

    gaps = np.concatenate(all_gaps)
    abs_gaps = np.abs(gaps)
    summary = {
        "status": "DEEPMATH_BASELINE_SAMPLED_TOKEN_LOGPROB_GAP_VIS_OK",
        "definition": "g_t = log pi_S(y_t|s_t) - log pi_T(y_t|s_t)",
        "student": args.student_label,
        "teacher": args.teacher_label,
        "rows": len(cases),
        "tokens": int(len(gaps)),
        "mean_gap": float(gaps.mean()),
        "mean_abs_gap": float(abs_gaps.mean()),
        "median_gap": float(np.median(gaps)),
        "positive_fraction": float((gaps > 0).mean()),
        "negative_fraction": float((gaps < 0).mean()),
        "abs_gap_quantiles": {str(q): float(np.quantile(abs_gaps, q)) for q in (0.5, 0.9, 0.95, 0.99)},
        "case_encoding": "gzip(base64(binary[n:uint32,scale:float32,(gap:int16,text_len:uint16,text:utf8)*n]))",
        "case_data_bytes": total_bytes,
    }
    index_payload = json.dumps({"summary": summary, "cases": cases}, ensure_ascii=False, separators=(",", ":"))
    template = args.template.read_text(encoding="utf-8")
    markers = {
        "__INDEX_JSON__": index_payload,
        "__PAGE_TITLE__": args.page_title,
        "__PAGE_HEADING__": args.page_heading,
    }
    for marker in markers:
        if template.count(marker) != 1:
            raise RuntimeError(f"Template must contain exactly one {marker} marker")
    for marker, value in markers.items():
        template = template.replace(marker, value)
    (args.output_dir / "index.html").write_text(template, encoding="utf-8")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
