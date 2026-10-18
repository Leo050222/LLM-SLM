#!/usr/bin/env python3
"""Render the per-bridge accuracy-lift histogram as a publication-ready PNG."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager, ticker


OUTPUT = Path("/Users/leolee/Documents/OPD/figures/bridge_accuracy_lift_distribution_20260902.png")

steps = list(range(-8, 9))
changes_pp = [step * 12.5 for step in steps]
counts = [
    88,
    174,
    298,
    756,
    1434,
    3079,
    6306,
    12485,
    43608,
    15912,
    9066,
    5134,
    3135,
    1744,
    1139,
    863,
    731,
]
total = sum(counts)

font_path = font_manager.findfont("Arial Unicode MS", fallback_to_default=False)
font_manager.fontManager.addfont(font_path)
plt.rcParams.update(
    {
        "font.family": "Arial Unicode MS",
        "axes.unicode_minus": False,
        "font.size": 12,
    }
)

negative_color = "#DE6B57"
neutral_color = "#7C8799"
positive_color = "#238A83"
colors = [
    negative_color if change < 0 else positive_color if change > 0 else neutral_color
    for change in changes_pp
]

fig, ax = plt.subplots(figsize=(14, 8), dpi=180)
fig.patch.set_facecolor("#F5F7FA")
ax.set_facecolor("#FFFFFF")

ax.axvspan(-106.25, -6.25, color=negative_color, alpha=0.045, linewidth=0)
ax.axvspan(6.25, 106.25, color=positive_color, alpha=0.045, linewidth=0)

bars = ax.bar(
    changes_pp,
    counts,
    width=9.5,
    color=colors,
    edgecolor="none",
    zorder=3,
)

for bar, count in zip(bars, counts):
    ax.annotate(
        f"{count:,}\n{count / total * 100:.2f}%",
        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
        xytext=(0, 6),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=9.2,
        color="#273142",
        linespacing=1.25,
        zorder=5,
    )

mean_lift = 4.939618884022954
ax.axvline(mean_lift, color="#303A4B", linewidth=1.35, linestyle=(0, (4, 4)), zorder=4)
ax.annotate(
    "平均提升 +4.94 pp",
    xy=(mean_lift, 41600),
    xytext=(13, 0),
    textcoords="offset points",
    ha="left",
    va="center",
    fontsize=10.5,
    color="#303A4B",
)

ax.set_title(
    "全部 105,952 个 Bridge 的正确率提升分布",
    loc="left",
    fontsize=20,
    fontweight="normal",
    color="#172033",
    pad=48,
)
ax.text(
    0,
    1.035,
    "下降 24,620（23.24%）     不变 43,608（41.16%）     提升 37,724（35.60%）",
    transform=ax.transAxes,
    ha="left",
    va="bottom",
    fontsize=12,
    color="#526075",
)

ax.set_xlabel("相对 direct Student 的正确率变化（百分点，pp）", labelpad=16, color="#273142")
ax.set_ylabel("Bridge 数量", labelpad=12, color="#273142")
ax.set_xlim(-106.25, 106.25)
ax.set_ylim(0, 48500)
ax.set_xticks(changes_pp)
ax.set_xticklabels(
    [
        "0" if value == 0 else f"{value:+g}"
        for value in changes_pp
    ],
    rotation=38,
    ha="right",
)
ax.yaxis.set_major_locator(ticker.MultipleLocator(10000))
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
ax.grid(axis="y", color="#DCE2EA", linewidth=0.8, alpha=0.85, zorder=0)
ax.grid(axis="x", visible=False)

for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color("#B8C1CE")
    ax.spines[side].set_linewidth(0.8)
ax.tick_params(axis="both", colors="#526075", length=0)

fig.text(
    0.075,
    0.025,
    "注：每个 bridge 有 8 次 continuation，因此变化以 12.5 pp 为一个档位；柱顶依次为数量和占比。",
    ha="left",
    va="bottom",
    fontsize=10,
    color="#68758A",
)

fig.subplots_adjust(left=0.075, right=0.985, bottom=0.17, top=0.82)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")
plt.close(fig)
print(OUTPUT)
