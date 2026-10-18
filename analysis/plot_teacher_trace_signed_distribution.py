from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "analysis" / "teacher_trace_fine_magnitude_counts_by_accuracy_lift_20260909.csv"
OUT = ROOT / "figures" / "teacher_trace_accuracy_lift_diverging_distribution_20260909.png"


def choose_font() -> font_manager.FontProperties:
    candidates = [
        Path("/Library/Fonts/Arial Unicode.ttf"),
        Path("/System/Library/Fonts/STHeiti Medium.ttc"),
        Path("/System/Library/Fonts/STHeiti Light.ttc"),
    ]
    for path in candidates:
        if path.exists():
            return font_manager.FontProperties(fname=str(path))
    return font_manager.FontProperties(family="DejaVu Sans")


FONT = choose_font()
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.unicode_minus": True,
    }
)


with DATA.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))

lifts = np.array([float(row["accuracy_lift_pp"]) for row in rows])
n = np.array([int(row["prefixes"]) for row in rows])

# Near zero -> far from zero, so magnitude increases away from the center line.
short_keys = [
    "short_0_100",
    "short_100_250",
    "short_250_500",
    "short_500_1000",
    "short_ge1000",
]
long_keys = [
    "long_0_100",
    "long_100_250",
    "long_250_500",
    "long_500_1000",
    "long_ge1000",
]
magnitude_labels = ["0–100", "100–250", "250–500", "500–1,000", "≥1,000"]

short = np.array([[int(row[key]) for key in short_keys] for row in rows], dtype=float)
long = np.array([[int(row[key]) for key in long_keys] for row in rows], dtype=float)
equal = np.array([int(row["equal"]) for row in rows], dtype=float)
short_pct = short / n[:, None] * 100
long_pct = long / n[:, None] * 100

# Colorblind-friendly blue/orange diverging palette. Darker colors mean larger changes.
short_colors = ["#cfe8f3", "#9ecae1", "#5fa6cf", "#2879b5", "#0b4f8a"]
long_colors = ["#fee2b7", "#fdc879", "#f4a340", "#df7627", "#a94714"]

fig = plt.figure(figsize=(15.2, 7.4), facecolor="white")
grid = fig.add_gridspec(1, 2, width_ratios=[2.35, 1.0], wspace=0.20)
ax = fig.add_subplot(grid[0, 0])
trend = fig.add_subplot(grid[0, 1])

y = np.arange(len(rows))
left_neg = np.zeros(len(rows))
left_pos = np.zeros(len(rows))

for idx, (label, color) in enumerate(zip(magnitude_labels, short_colors)):
    values = short_pct[:, idx]
    ax.barh(
        y,
        -values,
        left=left_neg,
        height=0.68,
        color=color,
        edgecolor="white",
        linewidth=0.55,
        label=f"缩短 {label}",
    )
    for yi, value, base in zip(y, values, left_neg):
        if value >= 6.6:
            ax.text(base - value / 2, yi, f"{value:.1f}", ha="center", va="center", fontsize=8.0, color="#17324d")
    left_neg -= values

for idx, (label, color) in enumerate(zip(magnitude_labels, long_colors)):
    values = long_pct[:, idx]
    ax.barh(
        y,
        values,
        left=left_pos,
        height=0.68,
        color=color,
        edgecolor="white",
        linewidth=0.55,
        label=f"增长 {label}",
    )
    for yi, value, base in zip(y, values, left_pos):
        if value >= 6.6:
            ax.text(base + value / 2, yi, f"{value:.1f}", ha="center", va="center", fontsize=8.0, color="#51240d")
    left_pos += values

ax.axvline(0, color="#273444", linewidth=1.25, zorder=5)
ax.set_xlim(-70, 70)
ax.set_xticks(np.arange(-60, 61, 20))
ax.set_xticklabels([f"{abs(int(value))}%" if value else "0" for value in np.arange(-60, 61, 20)], fontproperties=FONT)
ax.set_yticks(y)
ax.set_yticklabels([f"+{lift:g} pp   (n={count:,})" for lift, count in zip(lifts, n)], fontproperties=FONT)
ax.set_xlabel("Prefix 占比（左：Teacher 更短；右：Teacher 更长）", fontproperties=FONT, labelpad=9)
ax.set_ylabel("Teacher 相对 Student 的正确率提升", fontproperties=FONT, labelpad=10)
ax.grid(axis="x", color="#d7dce2", linewidth=0.7, alpha=0.72)
ax.set_axisbelow(True)
ax.invert_yaxis()
ax.text(-67.5, -0.83, "← Trace 缩短", ha="left", va="center", fontsize=11, fontproperties=FONT, color="#0b4f8a", weight="bold")
ax.text(67.5, -0.83, "Trace 增长 →", ha="right", va="center", fontsize=11, fontproperties=FONT, color="#a94714", weight="bold")

# Compact magnitude legend; direction is encoded by color family and position.
legend_items = [Patch(facecolor=color, edgecolor="none", label=label) for color, label in zip(short_colors, magnitude_labels)]
legend = ax.legend(
    handles=legend_items,
    title="颜色深浅：|Δ trace tokens|",
    ncol=5,
    loc="lower center",
    bbox_to_anchor=(0.5, 1.06),
    frameon=False,
    columnspacing=1.2,
    handlelength=1.5,
)
plt.setp(legend.get_title(), fontproperties=FONT, fontsize=9)
for text in legend.get_texts():
    text.set_fontproperties(FONT)

# Right panel: expose the most stable trend in the tails and show overall direction.
short_total = short_pct.sum(axis=1)
large_short = short_pct[:, 3:].sum(axis=1)
large_long = long_pct[:, 3:].sum(axis=1)

trend.plot(lifts, short_total, color="#2879b5", marker="o", linewidth=2.2, markersize=5.5, label="全部缩短")
trend.plot(lifts, large_short, color="#0b4f8a", marker="s", linewidth=2.0, markersize=5.0, label="缩短 ≥500")
trend.plot(lifts, large_long, color="#df7627", marker="s", linewidth=2.0, markersize=5.0, label="增长 ≥500")

for x, value in zip(lifts, short_total):
    trend.annotate(f"{value:.1f}%", (x, value), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8, color="#2879b5")

trend.set_xlim(7, 105.5)
trend.set_ylim(0, 58)
trend.set_xticks(lifts)
trend.set_xticklabels([f"+{lift:g}" for lift in lifts], rotation=35, ha="right", fontproperties=FONT)
trend.set_xlabel("正确率提升（pp）", fontproperties=FONT, labelpad=8)
trend.set_ylabel("Prefix 占比（%）", fontproperties=FONT, labelpad=8)
trend.grid(color="#d7dce2", linewidth=0.7, alpha=0.72)
trend.set_axisbelow(True)
trend_legend = trend.legend(loc="upper left", frameon=False)
for text in trend_legend.get_texts():
    text.set_fontproperties(FONT)

trend.annotate(
    "大幅增长减少",
    xy=(100, large_long[-1]),
    xytext=(67, 22),
    arrowprops={"arrowstyle": "->", "color": "#a94714", "lw": 1.1},
    color="#a94714",
    fontsize=9,
    fontproperties=FONT,
)
trend.annotate(
    "大幅缩短增加",
    xy=(100, large_short[-1]),
    xytext=(58, 5),
    arrowprops={"arrowstyle": "->", "color": "#0b4f8a", "lw": 1.1},
    color="#0b4f8a",
    fontsize=9,
    fontproperties=FONT,
)

fig.text(0.065, 0.955, "a  各提升档位的 trace 长度变化分布", fontsize=12, weight="bold", fontproperties=FONT, color="#1f2937")
fig.text(0.724, 0.955, "b  方向与尾部趋势", fontsize=12, weight="bold", fontproperties=FONT, color="#1f2937")
fig.text(
    0.065,
    0.018,
    "注：ΔL = Teacher continuation − Student continuation；仅统计 Teacher 正确率高于 Student 的 prefix。条内数字为该区间占比（%）。",
    fontsize=9,
    fontproperties=FONT,
    color="#4b5563",
)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight", facecolor="white", pad_inches=0.12)
plt.close(fig)
print(OUT)
