#!/usr/bin/env python3
"""Verify OPD token artifacts and analyze sampled reverse KL offline.

This script consumes already-generated student rollouts and chosen-token
log-probabilities.  It never loads a model, generates tokens, creates an
optimizer, calls backward, or mutates model parameters.

For a student-sampled token a, with

    gap = log p_T(a | s) - log p_S(a | s),

the two Monte Carlo estimators reported here are

    k1 = -gap
    k3 = exp(gap) - 1 - gap.

Their expectation over actions sampled from p_S is KL(p_S || p_T).  A single
k1 value can be negative.  Neither estimator is an exact full-vocabulary KL.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


METRICS = (
    "sampled_reverse_kl_k1_raw",
    "sampled_reverse_kl_k1_clipped",
    "sampled_reverse_kl_k3_raw",
    "sampled_reverse_kl_k3_clipped",
)
REQUIRED_TOKEN_COLUMNS = {
    "dataset_index",
    "problem_id",
    "correct",
    "response_length",
    "response_position",
    "relative_position",
    "token_id",
    "token_piece",
    "token_text",
    "student_chosen_logprob",
    "teacher_chosen_logprob",
    "chosen_logprob_gap",
}
QUANTILES = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 1.0)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")


def _shard_indices(paths: list[Path]) -> list[int]:
    indices: list[int] = []
    for path in paths:
        match = re.fullmatch(r"example_(\d+)\.(?:npz|parquet)", path.name)
        if match is None:
            raise ValueError(f"Unexpected shard name: {path.name}")
        indices.append(int(match.group(1)))
    return indices


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_artifacts(input_dir: Path, frame: pd.DataFrame) -> dict[str, Any]:
    """Perform structural and semantic checks without changing source artifacts."""
    required_files = [
        "rollouts.jsonl",
        "run_config.json",
        "teacher_manifest.json",
        "summary.json",
        "tokens.parquet",
    ]
    for name in required_files:
        _check((input_dir / name).is_file(), f"Missing required artifact: {name}")

    missing_columns = REQUIRED_TOKEN_COLUMNS.difference(frame.columns)
    _check(not missing_columns, f"tokens.parquet missing columns: {sorted(missing_columns)}")

    rollouts = read_jsonl(input_dir / "rollouts.jsonl")
    config = json.loads((input_dir / "run_config.json").read_text(encoding="utf-8"))
    manifest = json.loads((input_dir / "teacher_manifest.json").read_text(encoding="utf-8"))
    prior_summary = json.loads((input_dir / "summary.json").read_text(encoding="utf-8"))
    rollout_indices = [int(row["dataset_index"]) for row in rollouts]
    expected = set(rollout_indices)
    _check(len(expected) == len(rollouts), "Duplicate dataset_index in rollouts.jsonl")

    teacher_paths = sorted((input_dir / "teacher_topk").glob("example_*.npz"))
    student_paths = sorted((input_dir / "token_shards").glob("example_*.parquet"))
    _check(set(_shard_indices(teacher_paths)) == expected, "Teacher shard indices mismatch")
    _check(set(_shard_indices(student_paths)) == expected, "Student shard indices mismatch")
    _check(
        {int(row["dataset_index"]) for row in manifest} == expected,
        "teacher_manifest indices mismatch",
    )

    rollout_lengths = {int(row["dataset_index"]): int(row["response_tokens"]) for row in rollouts}
    token_lengths = frame.groupby("dataset_index", observed=True).size().astype(int).to_dict()
    _check(token_lengths == rollout_lengths, "Per-rollout token counts mismatch")
    _check(int(len(frame)) == sum(rollout_lengths.values()), "Combined token count mismatch")
    _check(int(prior_summary["num_tokens"]) == len(frame), "summary.json token count mismatch")
    _check(int(prior_summary["num_rollouts"]) == len(rollouts), "summary.json rollout count mismatch")

    numeric = frame[
        ["student_chosen_logprob", "teacher_chosen_logprob", "chosen_logprob_gap"]
    ].to_numpy(dtype=np.float64)
    _check(np.isfinite(numeric).all(), "Non-finite chosen-token log-probability data")
    recomputed_gap = numeric[:, 1] - numeric[:, 0]
    max_gap_error = float(np.max(np.abs(recomputed_gap - numeric[:, 2])))
    _check(max_gap_error <= 2e-6, f"chosen_logprob_gap mismatch: max error {max_gap_error}")

    position_errors: list[int] = []
    for dataset_index, group in frame.groupby("dataset_index", observed=True, sort=False):
        positions = group["response_position"].to_numpy(dtype=np.int64)
        if not np.array_equal(positions, np.arange(len(group), dtype=np.int64)):
            position_errors.append(int(dataset_index))
    _check(not position_errors, f"Non-contiguous response positions: {position_errors}")

    teacher_shapes: dict[int, list[int]] = {}
    teacher_nonfinite = 0
    for path in teacher_paths:
        dataset_index = _shard_indices([path])[0]
        with np.load(path) as data:
            _check(
                {"teacher_topk_ids", "teacher_topk_logprobs", "teacher_chosen_logprob"}
                <= set(data.files),
                f"Missing arrays in {path.name}",
            )
            ids = data["teacher_topk_ids"]
            topk_lp = data["teacher_topk_logprobs"]
            chosen_lp = data["teacher_chosen_logprob"]
            expected_length = rollout_lengths[dataset_index]
            _check(ids.shape == topk_lp.shape, f"Top-k shape mismatch in {path.name}")
            _check(ids.ndim == 2 and ids.shape[0] == expected_length, f"Bad shape in {path.name}")
            _check(chosen_lp.shape == (expected_length,), f"Bad chosen shape in {path.name}")
            teacher_nonfinite += int((~np.isfinite(topk_lp)).sum() + (~np.isfinite(chosen_lp)).sum())
            teacher_shapes[dataset_index] = [int(ids.shape[0]), int(ids.shape[1])]
    _check(teacher_nonfinite == 0, f"Teacher shards contain {teacher_nonfinite} non-finite values")

    sampling = config.get("sampling", {})
    expected_sampling = {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "do_sample": True,
    }
    for key, value in expected_sampling.items():
        _check(sampling.get(key) == value, f"Unexpected sampling setting {key}={sampling.get(key)!r}")
    safety_flags = {
        "parameters_frozen": config.get("parameters_frozen") is True,
        "optimizer_created_false": config.get("optimizer_created") is False,
        "backward_called_false": config.get("backward_called") is False,
        "student_parameters_require_grad_false": config.get("student_parameters_require_grad") is False,
    }
    _check(all(safety_flags.values()), f"Unsafe run_config flags: {safety_flags}")

    return {
        "status": "passed",
        "input_dir": str(input_dir),
        "num_rollouts": len(rollouts),
        "num_tokens": int(len(frame)),
        "teacher_shards": len(teacher_paths),
        "student_token_shards": len(student_paths),
        "dataset_indices": sorted(expected),
        "teacher_topk_widths": sorted({shape[1] for shape in teacher_shapes.values()}),
        "max_chosen_gap_recompute_abs_error": max_gap_error,
        "nonfinite_chosen_values": int((~np.isfinite(numeric)).sum()),
        "nonfinite_teacher_values": teacher_nonfinite,
        "sampling": sampling,
        "safety_flags": safety_flags,
    }


def add_estimators(frame: pd.DataFrame, clip_log_ratio: float) -> pd.DataFrame:
    """Return a compact token table with raw and reporting-only clipped estimators."""
    gap = frame["chosen_logprob_gap"].to_numpy(dtype=np.float64)
    k1_raw = -gap
    gap_clipped = np.clip(gap, -clip_log_ratio, clip_log_ratio)
    with np.errstate(over="ignore", invalid="ignore"):
        k3_raw = np.expm1(gap) - gap
        k3_clipped = np.expm1(gap_clipped) - gap_clipped
    result = frame[
        [
            "dataset_index",
            "problem_id",
            "correct",
            "response_length",
            "response_position",
            "relative_position",
            "token_id",
            "token_piece",
            "token_text",
            "student_chosen_logprob",
            "teacher_chosen_logprob",
            "chosen_logprob_gap",
        ]
    ].copy()
    result["sampled_reverse_kl_k1_raw"] = k1_raw
    result["sampled_reverse_kl_k1_clipped"] = np.clip(
        k1_raw, -clip_log_ratio, clip_log_ratio
    )
    result["sampled_reverse_kl_k3_raw"] = k3_raw
    result["sampled_reverse_kl_k3_clipped"] = k3_clipped
    return result


def describe(values: pd.Series) -> dict[str, Any]:
    array = values.to_numpy(dtype=np.float64)
    finite = array[np.isfinite(array)]
    result: dict[str, Any] = {
        "count": int(array.size),
        "finite_count": int(finite.size),
        "nonfinite_count": int(array.size - finite.size),
    }
    if finite.size == 0:
        return result
    quantiles = np.quantile(finite, QUANTILES)
    result.update(
        {
            "mean": float(np.mean(finite)),
            "std": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
            "sem": float(np.std(finite, ddof=1) / math.sqrt(finite.size))
            if finite.size > 1
            else 0.0,
            "negative_fraction": float(np.mean(finite < 0)),
            "zero_fraction": float(np.mean(finite == 0)),
            "quantiles": {
                str(level).replace(".", "_"): float(value)
                for level, value in zip(QUANTILES, quantiles, strict=True)
            },
        }
    )
    return result


def summarize_weighting(tokens: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    per_rollout = (
        tokens.groupby(["dataset_index", "problem_id", "correct"], observed=True)[list(METRICS)]
        .mean()
        .reset_index()
    )
    lengths = tokens.groupby("dataset_index", observed=True).size().rename("tokens")
    per_rollout = per_rollout.join(lengths, on="dataset_index")
    summary = {
        "token_weighted": {metric: describe(tokens[metric]) for metric in METRICS},
        "rollout_weighted": {metric: describe(per_rollout[metric]) for metric in METRICS},
    }
    return summary, per_rollout


def summarize_correctness(tokens: pd.DataFrame, per_rollout: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for correct in (False, True):
        token_group = tokens[tokens["correct"] == correct]
        rollout_group = per_rollout[per_rollout["correct"] == correct]
        if token_group.empty:
            continue
        result[str(correct).lower()] = {
            "tokens": int(len(token_group)),
            "rollouts": int(token_group["dataset_index"].nunique()),
            "token_weighted": {metric: describe(token_group[metric]) for metric in METRICS},
            "rollout_weighted": {metric: describe(rollout_group[metric]) for metric in METRICS},
        }
    return result


def position_table(tokens: pd.DataFrame, kind: str) -> pd.DataFrame:
    work = tokens.copy()
    if kind == "absolute":
        work["bin"] = (work["response_position"] // 256).astype(int)
        work["position_start"] = work["bin"] * 256
        work["position_end"] = work["position_start"] + 255
    elif kind == "relative":
        work["bin"] = np.minimum((work["relative_position"] * 20).astype(int), 19)
        work["position_start"] = work["bin"] / 20.0
        work["position_end"] = (work["bin"] + 1) / 20.0
    else:
        raise ValueError(kind)
    rows: list[dict[str, Any]] = []
    for bin_value, group in work.groupby("bin", observed=True, sort=True):
        row: dict[str, Any] = {
            "bin": int(bin_value),
            "position_start": float(group["position_start"].iloc[0]),
            "position_end": float(group["position_end"].iloc[0]),
            "tokens": int(len(group)),
            "active_rollouts": int(group["dataset_index"].nunique()),
        }
        for metric in METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_p50"] = float(group[metric].median())
            row[f"{metric}_p90"] = float(group[metric].quantile(0.9))
            row[f"{metric}_p99"] = float(group[metric].quantile(0.99))
        rows.append(row)
    return pd.DataFrame(rows)


def tail_table(tokens: pd.DataFrame) -> pd.DataFrame:
    series = {
        "abs_k1_raw": np.abs(tokens["sampled_reverse_kl_k1_raw"].to_numpy(dtype=np.float64)),
        "k3_raw": tokens["sampled_reverse_kl_k3_raw"].to_numpy(dtype=np.float64),
        "k3_clipped": tokens["sampled_reverse_kl_k3_clipped"].to_numpy(dtype=np.float64),
    }
    positive = np.concatenate([values[np.isfinite(values) & (values > 0)] for values in series.values()])
    if positive.size:
        lower = max(float(np.quantile(positive, 0.001)), 1e-12)
        upper = float(np.max(positive))
        thresholds = np.unique(np.concatenate(([0.0], np.geomspace(lower, upper, 160))))
    else:
        thresholds = np.array([0.0])
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        row = {"threshold": float(threshold)}
        for name, values in series.items():
            finite = values[np.isfinite(values)]
            row[f"{name}_ccdf_ge"] = float(np.mean(finite >= threshold))
        rows.append(row)
    return pd.DataFrame(rows)


def extreme_table(tokens: pd.DataFrame, count: int) -> pd.DataFrame:
    base_columns = [
        "dataset_index",
        "problem_id",
        "correct",
        "response_position",
        "relative_position",
        "token_id",
        "token_piece",
        "token_text",
        "student_chosen_logprob",
        "teacher_chosen_logprob",
        "chosen_logprob_gap",
        *METRICS,
    ]
    selections: list[pd.DataFrame] = []
    specs = {
        "largest_abs_k1": tokens["sampled_reverse_kl_k1_raw"].abs(),
        "largest_positive_k1": tokens["sampled_reverse_kl_k1_raw"],
        "most_negative_k1": -tokens["sampled_reverse_kl_k1_raw"],
        "largest_k3": tokens["sampled_reverse_kl_k3_raw"],
    }
    for category, ranking in specs.items():
        indices = ranking.nlargest(count).index
        selected = tokens.loc[indices, base_columns].copy()
        selected.insert(0, "extreme_category", category)
        selected.insert(1, "rank", np.arange(1, len(selected) + 1))
        selections.append(selected)
    return pd.concat(selections, ignore_index=True)


def create_figures(
    tokens: pd.DataFrame,
    per_rollout: pd.DataFrame,
    relative: pd.DataFrame,
    absolute: pd.DataFrame,
    tails: pd.DataFrame,
    output_dir: Path,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 140, "axes.grid": True, "grid.alpha": 0.25})
    figure_names: list[str] = []

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    k1 = tokens["sampled_reverse_kl_k1_clipped"].to_numpy(dtype=float)
    lo, hi = np.quantile(k1, [0.001, 0.999])
    axes[0, 0].hist(k1[(k1 >= lo) & (k1 <= hi)], bins=100, color="#3569b0")
    axes[0, 0].axvline(0, color="black", linewidth=1)
    axes[0, 0].set_title("k1 per-token distribution (0.1%–99.9%)")
    axes[0, 0].set_xlabel("log pS − log pT")

    for column, label in (
        ("abs_k1_raw_ccdf_ge", "|k1 raw|"),
        ("k3_raw_ccdf_ge", "k3 raw"),
        ("k3_clipped_ccdf_ge", "k3 clipped"),
    ):
        positive = tails["threshold"] > 0
        axes[0, 1].loglog(tails.loc[positive, "threshold"], tails.loc[positive, column], label=label)
    axes[0, 1].set_title("Token-weighted empirical CCDF")
    axes[0, 1].set_xlabel("threshold")
    axes[0, 1].set_ylabel("P(metric ≥ threshold)")
    axes[0, 1].legend()

    x = (relative["position_start"] + relative["position_end"]) / 2
    axes[1, 0].plot(x, relative["sampled_reverse_kl_k1_raw_mean"], marker="o", label="k1 raw")
    axes[1, 0].plot(x, relative["sampled_reverse_kl_k3_raw_mean"], marker="o", label="k3 raw")
    axes[1, 0].axhline(0, color="black", linewidth=1)
    axes[1, 0].set_title("Relative-position trend (token-weighted mean)")
    axes[1, 0].set_xlabel("relative response position")
    axes[1, 0].legend()

    categories = [False, True]
    x_values = np.arange(len(categories))
    width = 0.36
    token_means = [
        tokens.loc[tokens["correct"] == value, "sampled_reverse_kl_k1_raw"].mean()
        for value in categories
    ]
    rollout_means = [
        per_rollout.loc[per_rollout["correct"] == value, "sampled_reverse_kl_k1_raw"].mean()
        for value in categories
    ]
    axes[1, 1].bar(x_values - width / 2, token_means, width, label="token-weighted")
    axes[1, 1].bar(x_values + width / 2, rollout_means, width, label="rollout-weighted")
    axes[1, 1].set_xticks(x_values, ["incorrect", "correct"])
    axes[1, 1].set_title("Mean sampled reverse-KL k1 by correctness")
    axes[1, 1].legend()

    fig.suptitle("AIME24 step-0 sampled reverse-KL diagnostics", fontsize=15)
    fig.tight_layout()
    name = "sampled_reverse_kl_overview.png"
    fig.savefig(output_dir / name, bbox_inches="tight")
    plt.close(fig)
    figure_names.append(name)

    fig, ax1 = plt.subplots(figsize=(13, 5.5))
    ax1.plot(
        absolute["position_start"],
        absolute["sampled_reverse_kl_k1_raw_mean"],
        label="k1 raw mean",
        color="#3569b0",
    )
    ax1.plot(
        absolute["position_start"],
        absolute["sampled_reverse_kl_k3_raw_mean"],
        label="k3 raw mean",
        color="#d06b32",
    )
    ax1.set_xlabel("absolute response-token position (256-token bins)")
    ax1.set_ylabel("sampled estimator")
    ax2 = ax1.twinx()
    ax2.fill_between(
        absolute["position_start"],
        absolute["active_rollouts"],
        alpha=0.15,
        color="black",
        label="active rollouts",
    )
    ax2.set_ylabel("active rollouts")
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")
    ax1.set_title("Absolute-position trend and surviving rollout count")
    fig.tight_layout()
    name = "sampled_reverse_kl_absolute_position.png"
    fig.savefig(output_dir / name, bbox_inches="tight")
    plt.close(fig)
    figure_names.append(name)
    return figure_names


def create_html_report(
    output_dir: Path,
    summary: dict[str, Any],
    figures: list[str],
    extremes: pd.DataFrame,
) -> None:
    token_stats = summary["weighting"]["token_weighted"]
    rollout_stats = summary["weighting"]["rollout_weighted"]
    rows = []
    for metric in METRICS:
        rows.append(
            {
                "metric": metric,
                "token-weighted mean": token_stats[metric].get("mean"),
                "rollout-weighted mean": rollout_stats[metric].get("mean"),
                "token p50": token_stats[metric].get("quantiles", {}).get("0_5"),
                "token p99": token_stats[metric].get("quantiles", {}).get("0_99"),
            }
        )
    stats_html = pd.DataFrame(rows).to_html(index=False, float_format=lambda value: f"{value:.8g}")
    extreme_html = extremes.head(20).to_html(index=False, escape=True, float_format=lambda value: f"{value:.7g}")
    images = "\n".join(
        f'<figure><img src="{html.escape(name)}" alt="{html.escape(name)}"><figcaption>{html.escape(name)}</figcaption></figure>'
        for name in figures
    )
    caveat = html.escape(summary["estimator_scope"]["not_exact_full_vocab_kl"])
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Sampled reverse-KL diagnostics</title>
<style>body{{font:15px/1.5 system-ui,sans-serif;max-width:1200px;margin:2rem auto;padding:0 1rem;color:#18212b}}img{{max-width:100%}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border:1px solid #ccd3db;padding:.35rem;text-align:right}}th:first-child,td:first-child{{text-align:left}}.warning{{background:#fff4d6;border-left:5px solid #d59a00;padding:1rem}}code{{background:#eef1f4;padding:.1rem .25rem}}</style></head>
<body><h1>AIME24 step-0 sampled reverse-KL diagnostics</h1>
<p class="warning"><strong>Estimator scope:</strong> {caveat}</p>
<p><code>k1 = log pS − log pT = −chosen_logprob_gap</code>; <code>k3 = exp(chosen_logprob_gap) − 1 − chosen_logprob_gap</code>. A per-token k1 sample may be negative. Clipped values use |log-ratio| ≤ {summary['clip_log_ratio']} and are reporting-only robustness views.</p>
<h2>Core statistics</h2>{stats_html}
<h2>Visualizations</h2>{images}
<h2>Extreme-token preview</h2>{extreme_html}
<p>Machine-readable details: <code>summary.json</code>, <code>verification.json</code>, <code>per_rollout.csv</code>, position CSVs, <code>tail_ccdf.csv</code>, and <code>extreme_tokens.jsonl</code>.</p>
</body></html>"""
    (output_dir / "index.html").write_text(document, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--clip-log-ratio",
        type=float,
        default=20.0,
        help="Symmetric log-ratio clip used only for reporting-safe variants (default: 20)",
    )
    parser.add_argument("--extreme-count", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = (args.output_dir or input_dir / "sampled_reverse_kl").resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(input_dir)
    if args.clip_log_ratio <= 0 or not math.isfinite(args.clip_log_ratio):
        raise ValueError("--clip-log-ratio must be finite and positive")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(input_dir / "tokens.parquet")
    verification = verify_artifacts(input_dir, frame)
    tokens = add_estimators(frame, args.clip_log_ratio)
    _check(
        np.all(tokens["sampled_reverse_kl_k3_clipped"].to_numpy() >= -1e-12),
        "k3 clipped must be non-negative",
    )
    weighting, per_rollout = summarize_weighting(tokens)
    correctness = summarize_correctness(tokens, per_rollout)
    absolute = position_table(tokens, "absolute")
    relative = position_table(tokens, "relative")
    tails = tail_table(tokens)
    extremes = extreme_table(tokens, args.extreme_count)

    tokens.to_parquet(output_dir / "token_estimators.parquet", index=False, compression="zstd")
    per_rollout.to_csv(output_dir / "per_rollout.csv", index=False)
    absolute.to_csv(output_dir / "position_absolute_256.csv", index=False)
    relative.to_csv(output_dir / "position_relative_20.csv", index=False)
    tails.to_csv(output_dir / "tail_ccdf.csv", index=False)
    extremes.to_csv(output_dir / "extreme_tokens.csv", index=False)
    write_jsonl(output_dir / "extreme_tokens.jsonl", extremes.to_dict(orient="records"))
    write_json(output_dir / "verification.json", verification)

    summary = {
        "dataset": "aime24",
        "source": str(input_dir),
        "num_rollouts": verification["num_rollouts"],
        "num_tokens": verification["num_tokens"],
        "clip_log_ratio": args.clip_log_ratio,
        "estimator_scope": {
            "student_sampling_distribution": "temperature=1, top_p=1, top_k=-1, do_sample=true",
            "k1_definition": "log p_S - log p_T = -chosen_logprob_gap",
            "k3_definition": "exp(chosen_logprob_gap) - 1 - chosen_logprob_gap",
            "single_token_k1_can_be_negative": True,
            "expectation_statement": "E over student-policy sampled actions equals conditional KL(S||T); reported aggregates also average over visited rollout states",
            "not_exact_full_vocab_kl": "These are sampled on-policy estimators of reverse KL on visited states, not exact full-vocabulary KL(S||T).",
            "clipped_metrics": "Numerically bounded reporting-only variants; clipping changes the estimator and can introduce bias.",
        },
        "weighting": weighting,
        "by_correctness": correctness,
        "tail_reference_thresholds": {
            metric: {
                str(threshold).replace(".", "_"): float(
                    np.mean(tokens[metric].to_numpy(dtype=float) >= threshold)
                )
                for threshold in (0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0)
            }
            for metric in ("sampled_reverse_kl_k3_raw", "sampled_reverse_kl_k3_clipped")
        },
        "files": {},
    }
    figures = create_figures(tokens, per_rollout, relative, absolute, tails, output_dir)
    summary["files"] = {
        "token_estimators": "token_estimators.parquet",
        "per_rollout": "per_rollout.csv",
        "absolute_position": "position_absolute_256.csv",
        "relative_position": "position_relative_20.csv",
        "tail_ccdf": "tail_ccdf.csv",
        "extreme_tokens": "extreme_tokens.jsonl",
        "verification": "verification.json",
        "figures": figures,
        "html_report": "index.html",
    }
    write_json(output_dir / "summary.json", summary)
    create_html_report(output_dir, summary, figures, extremes)
    print(json.dumps({"status": "ok", "output_dir": str(output_dir), **verification}, default=_json_default))


if __name__ == "__main__":
    main()
