#!/usr/bin/env python3
"""Export an OPD token diagnostic parquet file as a reusable offline HTML viewer."""

from __future__ import annotations

import argparse
import base64
import gzip
import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "dataset_index",
    "problem_id",
    "correct",
    "response_position",
    "token_text",
    "raw_forward_kl_top128",
    "chosen_logprob_gap",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a self-contained, offline token-level KL heatmap page."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Diagnostic output directory or its tokens.parquet file.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title")
    parser.add_argument("--window-size", type=int, default=1024)
    parser.add_argument(
        "--template",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "tools"
        / "opd_token_heatmap"
        / "template.html",
    )
    return parser.parse_args()


def resolve_tokens_path(value: Path) -> Path:
    path = value.expanduser().resolve()
    if path.is_dir():
        path = path / "tokens.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def finite_cap(values: np.ndarray, *, absolute: bool = False) -> float:
    work = np.abs(values) if absolute else values
    finite = work[np.isfinite(work)]
    if finite.size == 0:
        return 1.0
    return max(float(np.max(finite)), np.finfo(np.float32).eps)


def color_cap(values: np.ndarray, *, absolute: bool = False) -> float:
    work = np.abs(values) if absolute else values
    finite = work[np.isfinite(work) & (work >= 0)]
    if finite.size == 0:
        return 1.0
    return max(float(np.quantile(finite, 0.999)), np.finfo(np.float32).eps)


def quantize_positive(values: np.ndarray, cap: float) -> np.ndarray:
    clean = np.nan_to_num(values, nan=0.0, posinf=cap, neginf=0.0)
    clean = np.clip(clean, 0.0, cap)
    return np.rint(255 * np.log1p(clean) / np.log1p(cap)).astype(np.uint8)


def quantize_signed(values: np.ndarray, cap: float) -> np.ndarray:
    clean = np.nan_to_num(values, nan=0.0, posinf=cap, neginf=-cap)
    magnitude = np.clip(np.abs(clean), 0.0, cap)
    quantized = np.rint(127 * np.log1p(magnitude) / np.log1p(cap)).astype(np.int16)
    signed = np.where(clean < 0, -quantized, quantized)
    return (128 + signed).astype(np.uint8)


def describe(values: np.ndarray, *, absolute_tail: bool = False) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"mean": 0.0, "p99": 0.0, "p99_9": 0.0, "min": 0.0, "max": 0.0}
    tail = np.abs(finite) if absolute_tail else finite
    return {
        "mean": float(np.mean(finite)),
        "p99": float(np.quantile(tail, 0.99)),
        "p99_9": float(np.quantile(tail, 0.999)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def gzip_b64(payload: bytes) -> str:
    return base64.b64encode(gzip.compress(payload, compresslevel=9)).decode("ascii")


def build_payload(frame: pd.DataFrame, source: Path) -> tuple[dict[str, Any], bytes]:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    frame = frame.sort_values(["dataset_index", "response_position"]).reset_index(drop=True)
    for dataset_index, group in frame.groupby("dataset_index", sort=True, observed=True):
        positions = group["response_position"].to_numpy(dtype=np.int64)
        if not np.array_equal(positions, np.arange(len(group), dtype=np.int64)):
            raise ValueError(f"Non-contiguous response positions for rollout {dataset_index}")

    texts = frame["token_text"].fillna("").astype(str).tolist()
    dictionary = list(dict.fromkeys(texts))
    dictionary_index = {value: index for index, value in enumerate(dictionary)}
    index_dtype = np.dtype("<u2") if len(dictionary) <= np.iinfo(np.uint16).max else np.dtype("<u4")
    token_indices = np.asarray([dictionary_index[value] for value in texts], dtype=index_dtype)

    forward = np.maximum(
        frame["raw_forward_kl_top128"].to_numpy(dtype=np.float64), 0.0
    )
    gap = frame["chosen_logprob_gap"].to_numpy(dtype=np.float64)
    k1 = -gap
    with np.errstate(over="ignore", invalid="ignore"):
        k3 = np.expm1(gap) - gap

    caps = {
        "fwd": finite_cap(forward),
        "k1": finite_cap(k1, absolute=True),
        "k3": finite_cap(k3),
    }
    color_caps = {
        "fwd": color_cap(forward),
        "k1": color_cap(k1, absolute=True),
        "k3": color_cap(k3),
    }
    quantized = np.stack(
        [
            quantize_positive(forward, caps["fwd"]),
            quantize_signed(k1, caps["k1"]),
            quantize_positive(k3, caps["k3"]),
        ],
        axis=1,
    )

    arrays = {"fwd": forward, "k1": k1, "k3": k3}
    rollouts: list[dict[str, Any]] = []
    offset = 0
    for dataset_index, group in frame.groupby("dataset_index", sort=True, observed=True):
        length = len(group)
        local = slice(offset, offset + length)
        rollouts.append(
            {
                "index": int(dataset_index),
                "problem": str(group["problem_id"].iloc[0]),
                "correct": bool(group["correct"].iloc[0]),
                "n": length,
                "offset": offset,
                "stats": {
                    name: describe(values[local], absolute_tail=(name == "k1"))
                    for name, values in arrays.items()
                },
            }
        )
        offset += length

    dataset = (
        str(frame["dataset"].iloc[0])
        if "dataset" in frame.columns and not frame.empty
        else source.parent.name
    )
    metadata: dict[str, Any] = {
        "version": 1,
        "dataset": dataset,
        "source": str(source),
        "total": int(len(frame)),
        "numRollouts": len(rollouts),
        "accuracy": float(frame.groupby("dataset_index", observed=True)["correct"].first().mean()),
        "dictionary": dictionary,
        "indexBytes": index_dtype.itemsize,
        "rollouts": rollouts,
        "caps": caps,
        "colorCaps": color_caps,
        "globalStats": {
            name: describe(values, absolute_tail=(name == "k1"))
            for name, values in arrays.items()
        },
        "layout": "token dictionary indices, then interleaved uint8 [forward, k1, k3]",
        "quantization": "8-bit log1p; k1 is signed around byte 128",
    }
    binary = token_indices.tobytes() + quantized.tobytes()
    return metadata, binary


def main() -> None:
    args = parse_args()
    if args.window_size <= 0:
        raise ValueError("--window-size must be positive")
    tokens_path = resolve_tokens_path(args.input)
    frame = pd.read_parquet(tokens_path)
    metadata, binary = build_payload(frame, tokens_path)

    template = args.template.expanduser().resolve().read_text(encoding="utf-8")
    title = args.title or f"{metadata['dataset']} rollout token KL heatmap"
    replacements = {
        "__PAGE_TITLE__": html.escape(title),
        "__WINDOW_SIZE__": str(args.window_size),
        "__META_B64__": gzip_b64(
            json.dumps(metadata, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ),
        "__DATA_B64__": gzip_b64(binary),
    }
    for marker, value in replacements.items():
        if template.count(marker) != 1:
            raise ValueError(f"Template must contain exactly one {marker}")
        template = template.replace(marker, value)

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template, encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "bytes": output.stat().st_size,
                "dataset": metadata["dataset"],
                "rollouts": metadata["numRollouts"],
                "tokens": metadata["total"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
