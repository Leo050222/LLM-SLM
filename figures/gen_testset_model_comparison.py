#!/usr/bin/env python3
"""Render the matched non-thinking x8 held-out comparison."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE / "testset_model_comparison_20260908.json").read_text())
METHODS = DATA["methods"]

plt.rcParams.update(
    {
        "font.size": 10,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]
labels = [
    "Student-only\n(initial)",
    "Traditional\nOPD\n(step 133)",
    "Bridge\nOPD\n(step 1656)",
    "Good-Bridge\nOPD\n(step 590)",
]
panels = [
    ("accuracy", "Rollout accuracy (Avg@8)", 0.33),
    ("pass_at_8", "Pass@8", 0.60),
    ("all_8_correct_rate", "All 8 correct", 0.105),
]

fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.8), constrained_layout=True)
for ax, (key, ylabel, ymax) in zip(axes, panels):
    vals = [m[key] for m in METHODS]
    bars = ax.bar(range(len(vals)), vals, color=colors, width=0.72)
    ax.set_xticks(range(len(vals)), labels)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, ymax)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + ymax * 0.018,
            f"{100 * val:.2f}%" if key == "accuracy" else f"{100 * val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="semibold",
        )

fig.text(
    0.5,
    -0.065,
    "Matched evaluation protocol — DeepMath GE6 test1000, non-thinking, 8 rollouts/question.\n"
    "Accuracy denominator: 8,000 rollouts/method; Pass@8 and all-8 denominator: 1,000 questions. Final-checkpoint training budgets differ.",
    ha="center",
    fontsize=9,
    color="#444444",
)

for suffix in ("png", "pdf"):
    fig.savefig(HERE / f"testset_model_comparison_20260908.{suffix}")
plt.close(fig)

print(HERE / "testset_model_comparison_20260908.png")
