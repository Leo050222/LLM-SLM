#!/usr/bin/env python3
"""Plot the full per-bridge accuracy-lift distribution for four horizons."""

from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DIST_CSV = ROOT / "analysis" / "bridge_lift_bucket_problem_dedup_20260906.csv"
PASS_CSV = ROOT / "analysis" / "bridge_pass_metrics_20260906.csv"
HORIZON_CSV = ROOT / "analysis" / "bridge_horizon_summary_20260906.csv"
OUTPUT_PNG = ROOT / "figures" / "bridge_lift_distribution_small_multiples_20260908.png"
N_BRIDGES = 105_952


def configure_fonts() -> None:
    for path in (
        Path("/System/Library/Fonts/STHeiti Medium.ttc"),
        Path("/System/Library/Fonts/PingFang.ttc"),
    ):
        if path.exists():
            fm.fontManager.addfont(path)
            plt.rcParams["font.family"] = fm.FontProperties(fname=path).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    configure_fonts()
    dist = pd.read_csv(DIST_CSV)
    pass_metrics = pd.read_csv(PASS_CSV).set_index("condition")
    horizon_summary = pd.read_csv(HORIZON_CSV).set_index("horizon_tokens")

    horizons = [32, 64, 128, 256]
    negative = "#D55E00"
    unchanged = "#7A7A7A"
    positive = "#0072B2"
    grid_color = "#E5E7EB"
    text_color = "#243041"
    muted = "#667085"

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.2), sharex=True, sharey=True)
    axes = axes.ravel()
    lifts = dist["lift_pp"].to_numpy()
    colors = [negative if value < 0 else unchanged if value == 0 else positive for value in lifts]

    for ax, horizon in zip(axes, horizons):
        counts = dist[f"h{horizon}_bridge_count"].to_numpy()
        percentages = counts / N_BRIDGES * 100
        bars = ax.barh(
            lifts,
            percentages,
            height=8.7,
            color=colors,
            alpha=0.88,
            edgecolor="none",
        )

        mean_lift = float(horizon_summary.loc[horizon, "lift"])
        row = pass_metrics.loc[f"{horizon}_token"]
        improved = float(row["improved_vs_direct_fraction_percent"])
        same = float(row["unchanged_vs_direct_fraction_percent"])
        degraded = float(row["degraded_vs_direct_fraction_percent"])

        ax.set_title(
            f"{horizon}-token bridge   平均提升 {mean_lift:+.2f} pp",
            fontsize=12.2,
            fontweight="bold",
            color=text_color,
            pad=9,
        )
        ax.text(
            0.02,
            0.02,
            f"提升 {improved:.1f}%   ·   持平 {same:.1f}%   ·   下降 {degraded:.1f}%",
            transform=ax.transAxes,
            fontsize=9.3,
            color=muted,
            ha="left",
            va="bottom",
        )

        for bar, value in zip(bars, percentages):
            if value >= 1.0:
                ax.text(
                    value + 0.45,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.1f}%",
                    va="center",
                    ha="left",
                    fontsize=7.6,
                    color=text_color,
                )

        ax.set_xlim(0, 49)
        ax.set_ylim(-106, 106)
        ax.set_xticks([0, 10, 20, 30, 40])
        ax.set_yticks([-100, -75, -50, -25, 0, 25, 50, 75, 100])
        ax.grid(axis="x", color=grid_color, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0, labelsize=9.2, colors=text_color)
        ax.tick_params(axis="x", labelsize=9.2, colors=muted)

    fig.suptitle(
        "Bridge 对 Prefix 正确率提升的完整分布",
        x=0.075,
        y=0.982,
        ha="left",
        fontsize=17,
        fontweight="bold",
        color="#111827",
    )
    fig.text(
        0.075,
        0.944,
        "每个面板包含 105,952 个 prefix–bridge；每个 bridge 基于 8 次 rollout，故提升档位间隔为 12.5 pp。",
        fontsize=10.3,
        color=muted,
        ha="left",
    )
    fig.supxlabel("该提升档位占全部 bridge 的比例 (%)", fontsize=11.5, y=0.047)
    fig.supylabel("相对同一 prefix 的 Student-only 正确率变化 (pp)", fontsize=11.5, x=0.018)

    legend_handles = [
        mpatches.Patch(color=negative, label="正确率下降"),
        mpatches.Patch(color=unchanged, label="持平"),
        mpatches.Patch(color=positive, label="正确率提升"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.955, 0.976),
        ncol=3,
        frameon=False,
        fontsize=9.5,
        handlelength=1.6,
        columnspacing=1.4,
    )
    fig.text(
        0.075,
        0.012,
        "条末仅标注占比 ≥1% 的档位；所有未标注的小档位仍按统一线性尺度绘制。",
        fontsize=9.0,
        color=muted,
        ha="left",
    )

    fig.subplots_adjust(left=0.10, right=0.965, top=0.89, bottom=0.10, wspace=0.16, hspace=0.19)
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUTPUT_PNG)


if __name__ == "__main__":
    main()
