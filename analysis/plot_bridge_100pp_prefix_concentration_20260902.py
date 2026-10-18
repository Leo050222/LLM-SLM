#!/usr/bin/env python3
"""Visualize concentration of +100 pp bridges across source prefixes."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager, ticker


OUTPUT = Path("/Users/leolee/Documents/OPD/figures/bridge_100pp_prefix_concentration_20260902.png")

multiplicity = list(range(1, 9))
prefix_counts = [117, 54, 32, 27, 10, 10, 8, 17]
bridge_counts = [117, 108, 96, 108, 50, 60, 56, 136]
prefix_total = sum(prefix_counts)
bridge_total = sum(bridge_counts)

font_path = font_manager.findfont("Arial Unicode MS", fallback_to_default=False)
font_manager.fontManager.addfont(font_path)
plt.rcParams.update(
    {
        "font.family": "Arial Unicode MS",
        "axes.unicode_minus": False,
        "font.size": 12,
    }
)

background = "#F5F7FA"
surface = "#FFFFFF"
foreground = "#172033"
muted = "#5D6A7E"
grid = "#DCE2EA"
prefix_color = "#405A78"
bridge_color = "#238A83"
highlight = "#7454A6"

fig, (ax1, ax2) = plt.subplots(
    2,
    1,
    figsize=(13, 8.5),
    dpi=180,
    sharex=True,
    gridspec_kw={"height_ratios": [1, 1], "hspace": 0.28},
)
fig.patch.set_facecolor(background)

for ax in (ax1, ax2):
    ax.set_facecolor(surface)
    ax.grid(axis="y", color=grid, linewidth=0.8, alpha=0.9, zorder=0)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#B8C1CE")
    ax.spines["bottom"].set_color("#B8C1CE")
    ax.tick_params(colors=muted, length=0)

prefix_colors = [prefix_color] * 7 + [highlight]
bridge_colors = [bridge_color] * 7 + [highlight]

bars1 = ax1.bar(multiplicity, prefix_counts, width=0.68, color=prefix_colors, edgecolor="none", zorder=3)
bars2 = ax2.bar(multiplicity, bridge_counts, width=0.68, color=bridge_colors, edgecolor="none", zorder=3)

for bar, value in zip(bars1, prefix_counts):
    ax1.annotate(
        f"{value}\n{value / prefix_total * 100:.2f}%",
        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
        xytext=(0, 6),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=10,
        color=foreground,
        linespacing=1.25,
    )

for bar, value in zip(bars2, bridge_counts):
    ax2.annotate(
        f"{value}\n{value / bridge_total * 100:.2f}%",
        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
        xytext=(0, 6),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=10,
        color=foreground,
        linespacing=1.25,
    )

ax1.set_ylim(0, 142)
ax2.set_ylim(0, 160)
ax1.yaxis.set_major_locator(ticker.MultipleLocator(25))
ax2.yaxis.set_major_locator(ticker.MultipleLocator(25))
ax1.set_ylabel("Prefix 数量", color=foreground, labelpad=14)
ax2.set_ylabel("贡献的 +100 pp Bridge 数量", color=foreground, labelpad=14)
ax2.set_xlabel("同一 Prefix 中达到 +100 pp 的 Bridge 数", color=foreground, labelpad=14)
ax2.set_xticks(multiplicity)
ax2.set_xticklabels([str(value) for value in multiplicity])

ax1.text(
    0.99,
    0.95,
    "117 个 Prefix 只命中 1 次",
    transform=ax1.transAxes,
    ha="right",
    va="top",
    fontsize=11,
    color=muted,
)
ax2.annotate(
    "17 个 Prefix 的 8 个 Bridge 全部达到 +100 pp\n贡献 136 个 Bridge（18.60%）",
    xy=(8, 136),
    xytext=(6.0, 146),
    ha="center",
    va="bottom",
    fontsize=10.5,
    color=foreground,
    arrowprops={"arrowstyle": "-", "color": "#7A879A", "linewidth": 1.0},
)

fig.suptitle(
    "+100 pp Bridge 明显集中于少数 Prefix",
    x=0.08,
    y=0.975,
    ha="left",
    fontsize=21,
    fontweight="normal",
    color=foreground,
)
fig.text(
    0.08,
    0.925,
    "731 个 Bridge 来自 275 个 Prefix；83.99% 来自重复命中的 Prefix",
    ha="left",
    va="top",
    fontsize=12.5,
    color=muted,
)
fig.text(
    0.08,
    0.025,
    "柱顶：数量与对应占比。上图占比以 275 个 Prefix 为分母；下图占比以 731 个 +100 pp Bridge 为分母。",
    ha="left",
    va="bottom",
    fontsize=10,
    color=muted,
)

fig.subplots_adjust(left=0.08, right=0.985, bottom=0.11, top=0.86)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")
plt.close(fig)
print(OUTPUT)
