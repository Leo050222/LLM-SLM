#!/usr/bin/env python3
"""Plot rollout-level average accuracy for student, bridge, and teacher runs."""

from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = ROOT / "analysis" / "prefix_bridge_teacher_average_accuracy_20260908.csv"
OUTPUT_PNG = ROOT / "figures" / "prefix_bridge_teacher_average_accuracy_20260908.png"


def configure_fonts() -> None:
    candidates = [
        Path("/System/Library/Fonts/STHeiti Medium.ttc"),
        Path("/System/Library/Fonts/PingFang.ttc"),
    ]
    for font_path in candidates:
        if font_path.exists():
            fm.fontManager.addfont(font_path)
            family = fm.FontProperties(fname=font_path).get_name()
            plt.rcParams["font.family"] = family
            break
    plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    configure_fonts()
    df = pd.read_csv(INPUT_CSV)

    labels = []
    for row in df.itertuples():
        if row.condition == "student_from_prefix":
            labels.append("Student from prefix")
        elif row.condition == "teacher_from_prefix":
            labels.append("Teacher from prefix")
        else:
            labels.append(f"Student + {row.bridge_tokens}-token bridge")
    df["label"] = labels

    baseline = float(
        df.loc[df["condition"] == "student_from_prefix", "average_accuracy_percent"].iloc[0]
    )
    y = list(range(len(df) - 1, -1, -1))

    colors = {
        "student_from_prefix": "#4B5563",
        "student_after_bridge": "#0072B2",
        "teacher_from_prefix": "#D55E00",
    }
    markers = {
        "student_from_prefix": "s",
        "student_after_bridge": "o",
        "teacher_from_prefix": "D",
    }

    fig = plt.figure(figsize=(11.6, 6.8), facecolor="white")
    grid = fig.add_gridspec(1, 2, width_ratios=[3.35, 1.65], wspace=0.04)
    ax = fig.add_subplot(grid[0, 0])
    ax_table = fig.add_subplot(grid[0, 1], sharey=ax)

    for yi, row in zip(y, df.itertuples()):
        color = colors[row.condition]
        ax.plot(
            [baseline, row.average_accuracy_percent],
            [yi, yi],
            color=color,
            linewidth=3.0 if row.condition == "student_after_bridge" else 2.2,
            alpha=0.72,
            solid_capstyle="round",
            zorder=2,
        )
        ax.scatter(
            row.average_accuracy_percent,
            yi,
            s=105 if row.condition == "student_after_bridge" else 120,
            marker=markers[row.condition],
            color=color,
            edgecolor="white",
            linewidth=1.2,
            zorder=3,
        )
        ax.text(
            row.average_accuracy_percent + 0.35,
            yi,
            f"{row.average_accuracy_percent:.2f}%",
            va="center",
            ha="left",
            fontsize=11.5,
            fontweight="bold",
            color=color,
        )

    ax.axvline(baseline, color="#6B7280", linestyle=(0, (4, 4)), linewidth=1.15, zorder=1)
    ax.text(
        baseline + 0.15,
        len(df) - 0.27,
        f"Student 基线 {baseline:.2f}%",
        fontsize=9.5,
        color="#4B5563",
        va="top",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(df["label"], fontsize=11)
    ax.set_xlim(37.5, 61.2)
    ax.set_ylim(-0.7, len(df) - 0.15)
    ax.set_xlabel("Rollout-level average accuracy (%)", fontsize=11.5, labelpad=10)
    ax.set_xticks([40, 45, 50, 55, 60])
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.9)
    ax.grid(axis="y", visible=False)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=10)
    ax.tick_params(axis="x", colors="#4B5563")

    ax_table.set_xlim(0, 1)
    ax_table.set_xticks([])
    ax_table.tick_params(axis="y", left=False, labelleft=False)
    ax_table.spines[:].set_visible(False)
    ax_table.set_facecolor("white")

    ax_table.text(0.03, len(df) - 0.15, "正确 / Rollouts", fontsize=10.5,
                  fontweight="bold", color="#374151", va="bottom")
    ax_table.text(0.97, len(df) - 0.15, "较 Student 提升", fontsize=10.5,
                  fontweight="bold", color="#374151", ha="right", va="bottom")
    for yi, row in zip(y, df.itertuples()):
        ax_table.text(
            0.03,
            yi,
            f"{row.correct_rollouts:,.0f} / {row.rollouts:,.0f}",
            va="center",
            fontsize=10.5,
            color="#374151",
        )
        delta = row.delta_vs_student_from_prefix_pp
        delta_text = "—" if abs(delta) < 1e-10 else f"+{delta:.2f} pp"
        ax_table.text(
            0.97,
            yi,
            delta_text,
            va="center",
            ha="right",
            fontsize=10.5,
            fontweight="bold" if delta > 0 else "normal",
            color=colors[row.condition] if delta > 0 else "#6B7280",
        )
        ax_table.axhline(yi - 0.5, color="#F1F5F9", linewidth=0.8, xmin=0.02, xmax=0.98)

    fig.suptitle(
        "不同 Bridge 支持力度下的平均正确率",
        x=0.075,
        y=0.97,
        ha="left",
        fontsize=17,
        fontweight="bold",
        color="#111827",
    )
    fig.text(
        0.075,
        0.915,
        "点表示正确率；连线相对 Student from prefix 基线。原始正确数仅作样本量说明，不直接比较。",
        fontsize=10.5,
        color="#4B5563",
        ha="left",
    )
    fig.text(
        0.075,
        0.025,
        "注：所有正确率均按 rollout-level average accuracy 计算；after-bridge 的 rollout 总量约为 from-prefix 的 8 倍。",
        fontsize=9.5,
        color="#6B7280",
        ha="left",
    )

    fig.subplots_adjust(left=0.27, right=0.97, top=0.86, bottom=0.14)
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUTPUT_PNG)


if __name__ == "__main__":
    main()
