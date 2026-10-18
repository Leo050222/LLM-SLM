#!/usr/bin/env python3
"""Visualize trace-length change across bridge accuracy-lift buckets."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager, ticker


OUTPUT = Path("/Users/leolee/Documents/OPD/figures/accuracy_lift_vs_trace_change_20260902.png")

lift_pp = [value * 12.5 for value in range(-8, 9)]
counts = [88, 174, 298, 756, 1434, 3079, 6306, 12485, 43608, 15912, 9066, 5134, 3135, 1744, 1139, 863, 731]
clean_mean = [-71.22, -4.06, -41.10, -61.75, -67.63, -60.27, -52.17, -30.78, -22.50, 3.00, 39.13, 68.20, 108.98, 143.68, 139.02, 141.98, 118.33]
clean_median = [-78.75, -27.25, -18.75, -51.75, -56.38, -60.50, -53.38, -41.50, -32.00, -5.63, 26.13, 42.50, 70.25, 102.63, 77.75, 73.50, 24.50]
shorter_pct = [32.93, 46.24, 46.31, 41.80, 41.59, 39.36, 40.18, 42.21, 43.90, 48.82, 54.58, 57.51, 60.51, 63.83, 61.36, 63.89, 54.10]

font_path = font_manager.findfont("Arial Unicode MS", fallback_to_default=False)
font_manager.fontManager.addfont(font_path)
plt.rcParams.update(
    {
        "font.family": "Arial Unicode MS",
        "axes.unicode_minus": False,
        "font.size": 11.5,
    }
)

background = "#F5F7FA"
surface = "#FFFFFF"
foreground = "#172033"
muted = "#5D6A7E"
grid = "#DCE2EA"
positive = "#238A83"
negative = "#DE6B57"
median_color = "#405A78"
count_color = "#8A96A8"

fig = plt.figure(figsize=(14, 10), dpi=180, facecolor=background)
gs = fig.add_gridspec(3, 1, height_ratios=[3.2, 1.55, 1.15], hspace=0.28)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharex=ax1)
ax3 = fig.add_subplot(gs[2], sharex=ax1)

for ax in (ax1, ax2, ax3):
    ax.set_facecolor(surface)
    ax.grid(axis="y", color=grid, linewidth=0.8, alpha=0.9, zorder=0)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#B8C1CE")
    ax.spines["bottom"].set_color("#B8C1CE")
    ax.tick_params(colors=muted, length=0)

ax1.axhspan(-110, 0, color=negative, alpha=0.05, linewidth=0)
ax1.axhspan(0, 180, color=positive, alpha=0.05, linewidth=0)
ax1.axhline(0, color="#68758A", linewidth=1.1, zorder=2)
ax1.plot(
    lift_pp,
    clean_mean,
    color=positive,
    linewidth=2.6,
    marker="o",
    markersize=6.5,
    label="平均缩短量",
    zorder=4,
)
ax1.plot(
    lift_pp,
    clean_median,
    color=median_color,
    linewidth=2.0,
    marker="D",
    markersize=5.0,
    label="中位缩短量",
    zorder=4,
)
ax1.set_ylim(-105, 175)
ax1.set_ylabel("完整 trace 缩短量（tokens）", color=foreground, labelpad=12)
ax1.legend(loc="upper left", frameon=False, ncols=2, labelcolor=foreground)
ax1.text(
    0.985,
    0.96,
    "档位均值相关：Pearson r = 0.897\n单 bridge 稳健相关：r ≈ 0.13",
    transform=ax1.transAxes,
    ha="right",
    va="top",
    color=muted,
    fontsize=10.5,
)
ax1.annotate(
    "+25 pp 后，均值与中位数均转为明显缩短",
    xy=(25, 39.13),
    xytext=(-5, 105),
    arrowprops={"arrowstyle": "-", "color": "#7A879A", "linewidth": 1.0},
    color=foreground,
    fontsize=10.5,
)

bar_colors = [positive if value >= 50 else negative for value in shorter_pct]
ax2.bar(lift_pp, shorter_pct, width=9.5, color=bar_colors, edgecolor="none", zorder=3)
ax2.axhline(50, color="#68758A", linewidth=1.1, linestyle=(0, (4, 4)), zorder=4)
ax2.set_ylim(28, 69)
ax2.set_ylabel("Trace 变短比例", color=foreground, labelpad=12)
ax2.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=0))
for x, value in zip(lift_pp, shorter_pct):
    ax2.text(x, value + 1.0, f"{value:.1f}%", ha="center", va="bottom", fontsize=8.5, color=foreground)

ax3.bar(lift_pp, counts, width=9.5, color=count_color, edgecolor="none", zorder=3)
ax3.set_yscale("log")
ax3.set_ylim(55, 80000)
ax3.set_ylabel("Bridge 数\n（对数轴）", color=foreground, labelpad=12)
ax3.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=4))
ax3.yaxis.set_minor_locator(ticker.NullLocator())
for x, value in zip(lift_pp, counts):
    ax3.text(x, value * 1.18, f"{value:,}", ha="center", va="bottom", fontsize=8.2, color=foreground)

labels = ["0" if value == 0 else f"{value:+g}" for value in lift_pp]
ax3.set_xticks(lift_pp)
ax3.set_xticklabels(labels, rotation=38, ha="right")
ax3.set_xlabel("相对 direct Student 的正确率变化（百分点，pp）", color=foreground, labelpad=14)
ax3.set_xlim(-106.25, 106.25)
plt.setp(ax1.get_xticklabels(), visible=False)
plt.setp(ax2.get_xticklabels(), visible=False)

fig.suptitle(
    "正确率提升越大，Bridge 后的完整 Trace 越倾向缩短",
    x=0.075,
    y=0.975,
    ha="left",
    fontsize=20,
    fontweight="normal",
    color=foreground,
)
fig.text(
    0.075,
    0.937,
    "按正确率变化档位聚合；均值/中位数剔除 length-finish；缩短量为正表示 after-bridge trace 更短",
    ha="left",
    va="top",
    fontsize=11.5,
    color=muted,
)
fig.text(
    0.075,
    0.018,
    "注：档位层面的趋势很强，但单个 bridge 的相关性较弱；两者的尺度不能混同。",
    ha="left",
    va="bottom",
    fontsize=10,
    color=muted,
)

fig.subplots_adjust(left=0.075, right=0.985, bottom=0.105, top=0.885)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")
plt.close(fig)
print(OUTPUT)
