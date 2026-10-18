#!/usr/bin/env python3
"""Analyze the completed 256-token anchor-bifurcation judge experiment."""

from __future__ import annotations

import csv
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/anchor_bifurcation_h256_openlux_luna_seed20260908/full"
OUT = RUN / "analysis"
FIGURE = ROOT / "figures/anchor_bifurcation_h256_openlux_analysis_20260908.png"
SEED = 20260908
BOOTSTRAPS = 10_000


def normal_two_sided_p(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2.0))


def ols_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    x0 = x - x.mean()
    slope = float(np.dot(x0, y - y.mean()) / np.dot(x0, x0))
    intercept = float(y.mean() - slope * x.mean())
    residual = y - (intercept + slope * x)
    s2 = float(np.dot(residual, residual) / (len(x) - 2))
    se = math.sqrt(s2 / float(np.dot(x0, x0)))
    r = float(np.corrcoef(x, y)[0, 1])
    return intercept, slope, se, r


def percentile_ci(values: np.ndarray) -> tuple[float, float]:
    lo, hi = np.quantile(values, [0.025, 0.975])
    return float(lo), float(hi)


def main() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/opd-mpl")
    OUT.mkdir(parents=True, exist_ok=True)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for path in sorted((RUN / "items").glob("*.json")):
        if path.name.endswith(".failed.json"):
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") != "ok":
            raise RuntimeError(f"non-ok item: {path}")
        judgment = record["judgment"]
        rows.append(
            {
                "item_id": record["item_id"],
                "dataset_index": int(record["dataset_index"]),
                "bridge_index": int(record["bridge_index"]),
                "lift_steps": int(record["lift_steps"]),
                "lift_pp": float(record["lift_pp"]),
                "outcome_cell": record["outcome_cell"],
                "direct_path_label": record["direct_path_label"],
                "topic": record.get("topic"),
                "difficulty": record.get("difficulty"),
                "anchor_yes": judgment["anchor_verdict"] == "yes",
                "anchor_uncertain": judgment["anchor_verdict"] == "uncertain",
                "direction_yes": judgment["reasoning_direction_changed"] == "yes",
                "immediate": judgment["divergence_onset"] == "immediately_after_anchor",
                "within_256": judgment["divergence_onset"] == "within_256_span",
                "after_256": judgment["divergence_onset"] == "after_256",
                "persistent": judgment["persistence"] == "persistent",
                "confidence": int(judgment["confidence"]),
                "change_types": judgment["change_types"],
            }
        )

    if len(rows) != 7200:
        raise RuntimeError(f"expected 7200 items, found {len(rows)}")

    grouped: dict[tuple[int, int, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["lift_steps"], row["dataset_index"], row["bridge_index"])].append(row)
    if len(grouped) != 900 or any(len(v) != 8 for v in grouped.values()):
        raise RuntimeError("expected 900 question-bridge groups with 8 judgments each")

    question_rows: list[dict] = []
    for (lift, dataset_index, bridge_index), values in sorted(grouped.items()):
        yes_count = sum(v["anchor_yes"] for v in values)
        question_rows.append(
            {
                "lift_steps": lift,
                "lift_pp": lift * 12.5,
                "dataset_index": dataset_index,
                "bridge_index": bridge_index,
                "anchor_yes_count": yes_count,
                "anchor_yes_rate": yes_count / 8,
                "majority_anchor_yes": yes_count >= 5,
                "unanimous_anchor_yes": yes_count == 8,
                "any_anchor_yes": yes_count >= 1,
                "direction_yes_rate": sum(v["direction_yes"] for v in values) / 8,
                "immediate_rate": sum(v["immediate"] for v in values) / 8,
                "persistent_rate": sum(v["persistent"] for v in values) / 8,
                "mean_confidence": sum(v["confidence"] for v in values) / 8,
            }
        )

    by_lift_q: dict[int, list[dict]] = defaultdict(list)
    by_lift_pair: dict[int, list[dict]] = defaultdict(list)
    for row in question_rows:
        by_lift_q[row["lift_steps"]].append(row)
    for row in rows:
        by_lift_pair[row["lift_steps"]].append(row)

    rng = np.random.default_rng(SEED)
    boot_rates = np.empty((BOOTSTRAPS, 9), dtype=float)
    boot_majority = np.empty((BOOTSTRAPS, 9), dtype=float)
    boot_slopes = np.empty(BOOTSTRAPS, dtype=float)
    x_question = np.array([q["lift_steps"] for q in question_rows], dtype=float)
    y_question = np.array([q["anchor_yes_rate"] for q in question_rows], dtype=float)
    _, slope, slope_se, pearson_r = ols_slope(x_question, y_question)
    for b in range(BOOTSTRAPS):
        bx: list[float] = []
        by: list[float] = []
        for lift in range(9):
            qs = by_lift_q[lift]
            idx = rng.integers(0, len(qs), size=len(qs))
            selected = [qs[i] for i in idx]
            rates = np.array([q["anchor_yes_rate"] for q in selected])
            major = np.array([q["majority_anchor_yes"] for q in selected], dtype=float)
            boot_rates[b, lift] = rates.mean()
            boot_majority[b, lift] = major.mean()
            bx.extend([float(lift)] * len(selected))
            by.extend(rates.tolist())
        boot_slopes[b] = ols_slope(np.array(bx), np.array(by))[1]

    summary_rows: list[dict] = []
    for lift in range(9):
        pairs = by_lift_pair[lift]
        qs = by_lift_q[lift]
        pair_yes = float(np.mean([r["anchor_yes"] for r in pairs]))
        majority = float(np.mean([q["majority_anchor_yes"] for q in qs]))
        rate_lo, rate_hi = percentile_ci(boot_rates[:, lift])
        maj_lo, maj_hi = percentile_ci(boot_majority[:, lift])
        summary_rows.append(
            {
                "lift_steps": lift,
                "lift_pp": lift * 12.5,
                "questions": len(qs),
                "judgments": len(pairs),
                "anchor_yes": sum(r["anchor_yes"] for r in pairs),
                "anchor_yes_rate": pair_yes,
                "anchor_yes_cluster_ci_low": rate_lo,
                "anchor_yes_cluster_ci_high": rate_hi,
                "majority_yes_questions": sum(q["majority_anchor_yes"] for q in qs),
                "majority_yes_rate": majority,
                "majority_yes_cluster_ci_low": maj_lo,
                "majority_yes_cluster_ci_high": maj_hi,
                "unanimous_yes_questions": sum(q["unanimous_anchor_yes"] for q in qs),
                "unanimous_yes_rate": float(np.mean([q["unanimous_anchor_yes"] for q in qs])),
                "direction_changed_rate": float(np.mean([r["direction_yes"] for r in pairs])),
                "immediate_divergence_rate": float(np.mean([r["immediate"] for r in pairs])),
                "persistent_rate": float(np.mean([r["persistent"] for r in pairs])),
                "mean_confidence": float(np.mean([r["confidence"] for r in pairs])),
            }
        )

    # Diagnostics for selected-pair outcome cells and blinded A/B order.
    outcome = {}
    for cell in ("F_to_F", "F_to_S", "S_to_F", "S_to_S"):
        subset = [r for r in rows if r["outcome_cell"] == cell]
        outcome[cell] = {
            "n": len(subset),
            "anchor_yes": sum(r["anchor_yes"] for r in subset),
            "anchor_yes_rate": float(np.mean([r["anchor_yes"] for r in subset])) if subset else None,
        }
    path_order = {}
    for label in ("A", "B"):
        subset = [r for r in rows if r["direct_path_label"] == label]
        path_order[label] = {
            "n": len(subset),
            "anchor_yes_rate": float(np.mean([r["anchor_yes"] for r in subset])),
        }

    change_counts = Counter(t for r in rows for t in r["change_types"])
    verdict_counts = Counter("yes" if r["anchor_yes"] else "uncertain" if r["anchor_uncertain"] else "no" for r in rows)
    slope_ci = percentile_ci(boot_slopes)
    endpoint_pair_ci = percentile_ci(boot_rates[:, 8] - boot_rates[:, 0])
    endpoint_majority_ci = percentile_ci(boot_majority[:, 8] - boot_majority[:, 0])
    z = slope / slope_se
    overall = {
        "judgments": len(rows),
        "question_bridge_groups": len(question_rows),
        "anchor_verdict_counts": dict(verdict_counts),
        "anchor_yes_rate": float(np.mean([r["anchor_yes"] for r in rows])),
        "majority_yes_questions": sum(q["majority_anchor_yes"] for q in question_rows),
        "majority_yes_rate": float(np.mean([q["majority_anchor_yes"] for q in question_rows])),
        "unanimous_yes_questions": sum(q["unanimous_anchor_yes"] for q in question_rows),
        "unanimous_yes_rate": float(np.mean([q["unanimous_anchor_yes"] for q in question_rows])),
        "direction_changed_rate": float(np.mean([r["direction_yes"] for r in rows])),
        "immediate_divergence_rate": float(np.mean([r["immediate"] for r in rows])),
        "persistent_rate": float(np.mean([r["persistent"] for r in rows])),
        "mean_confidence": float(np.mean([r["confidence"] for r in rows])),
        "lift_linear_slope_per_step": slope,
        "lift_linear_slope_pp_per_12_5pp": slope * 100,
        "lift_linear_slope_bootstrap_ci": list(slope_ci),
        "lift_linear_slope_normal_p": normal_two_sided_p(z),
        "lift_question_level_pearson_r": pearson_r,
        "endpoint_pair_rate_difference": summary_rows[-1]["anchor_yes_rate"] - summary_rows[0]["anchor_yes_rate"],
        "endpoint_pair_rate_difference_cluster_ci": list(endpoint_pair_ci),
        "endpoint_majority_rate_difference": summary_rows[-1]["majority_yes_rate"] - summary_rows[0]["majority_yes_rate"],
        "endpoint_majority_rate_difference_cluster_ci": list(endpoint_majority_ci),
        "outcome_cells": outcome,
        "path_order": path_order,
        "path_order_rate_difference_A_minus_B": path_order["A"]["anchor_yes_rate"] - path_order["B"]["anchor_yes_rate"],
        "change_type_counts": dict(change_counts.most_common()),
    }

    with (OUT / "by_lift.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    with (OUT / "question_level.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(question_rows[0]))
        writer.writeheader()
        writer.writerows(question_rows)
    (OUT / "analysis_summary.json").write_text(
        json.dumps({"overall": overall, "by_lift": summary_rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lift = np.arange(9)
    x_pp = lift * 12.5
    rate = np.array([r["anchor_yes_rate"] for r in summary_rows])
    rate_lo = np.array([r["anchor_yes_cluster_ci_low"] for r in summary_rows])
    rate_hi = np.array([r["anchor_yes_cluster_ci_high"] for r in summary_rows])
    majority = np.array([r["majority_yes_rate"] for r in summary_rows])
    unanimous = np.array([r["unanimous_yes_rate"] for r in summary_rows])

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=180)
    ax = axes[0, 0]
    ax.plot(x_pp, rate * 100, marker="o", lw=2.4, color="#2457C5", label="Pair-level yes")
    ax.fill_between(x_pp, rate_lo * 100, rate_hi * 100, color="#2457C5", alpha=0.16, label="95% cluster bootstrap CI")
    ax.set_title("Strict logical-bifurcation verdict")
    ax.set_xlabel("Bridge success-rate lift (percentage points)")
    ax.set_ylabel("Anchor = yes (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[0, 1]
    ax.plot(x_pp, majority * 100, marker="o", lw=2.2, label="Majority yes (>=5/8)", color="#008A78")
    ax.plot(x_pp, unanimous * 100, marker="s", lw=2.2, label="Unanimous yes (8/8)", color="#D45B33")
    ax.set_title("Question-level stability across 8 trajectory pairs")
    ax.set_xlabel("Bridge success-rate lift (percentage points)")
    ax.set_ylabel("Questions (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1, 0]
    ax.plot(x_pp, [r["direction_changed_rate"] * 100 for r in summary_rows], marker="o", label="Direction changed", color="#6C4CCF")
    ax.plot(x_pp, [r["immediate_divergence_rate"] * 100 for r in summary_rows], marker="s", label="Immediate divergence", color="#E09526")
    ax.plot(x_pp, [r["persistent_rate"] * 100 for r in summary_rows], marker="^", label="Persistent effect", color="#3B8F47")
    ax.set_title("Components of the strict verdict")
    ax.set_xlabel("Bridge success-rate lift (percentage points)")
    ax.set_ylabel("Judgments (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1, 1]
    cells = list(outcome)
    vals = [outcome[c]["anchor_yes_rate"] * 100 for c in cells]
    bars = ax.bar(cells, vals, color=["#9CA3AF", "#2CA25F", "#D95F5F", "#4C78A8"])
    ax.set_title("Verdict by paired rollout outcome")
    ax.set_ylabel("Anchor = yes (%)")
    ax.set_ylim(0, 100)
    for bar, value in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.4, f"{value:.1f}%", ha="center", fontsize=9)

    fig.suptitle("256-token Bridge: Is the anchor a genuine logical bifurcation?", fontsize=16, fontweight="bold")
    fig.text(0.5, 0.01, "900 question–bridge groups; 8 blinded trajectory-pair judgments each; OpenLux gpt-5.6-luna", ha="center", fontsize=9, color="#555")
    fig.tight_layout(rect=[0, 0.035, 1, 0.96])
    fig.savefig(FIGURE, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps({"overall": overall, "by_lift": summary_rows, "figure": str(FIGURE)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
