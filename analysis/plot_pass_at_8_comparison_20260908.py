#!/usr/bin/env python3
"""Visualize pass@8 across direct, bridge-supported, and teacher rollouts."""

from pathlib import Path
import json

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BRIDGE_CSV = ROOT / "analysis" / "bridge_pass_metrics_20260906.csv"
ROLLOUT_JSON = ROOT / "figures" / "rollout_comparison_png" / "rollout_metrics.json"
OUTPUT_PNG = ROOT / "figures" / "pass_at_8_comparison_20260908.png"


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


def load_data() -> pd.DataFrame:
    bridge = pd.read_csv(BRIDGE_CSV)
    with ROLLOUT_JSON.open("r", encoding="utf-8") as handle:
        rollout = json.load(handle)["experiments"]
    teacher = next(row for row in rollout if row["name"].replace("\n", " ") == "Teacher from prefix")

    rows = []
    for row in bridge.itertuples():
        if row.condition == "direct_student":
            label = "Student from prefix"
            kind = "student"
            total_rollouts = row.group_count * 8
        else:
            tokens = row.condition.split("_")[0]
            label = f"Student + {tokens}-token bridge"
            kind = "bridge"
            total_rollouts = row.group_count * 8
        rows.append(
            {
                "label": label,
                "kind": kind,
                "units": int(row.group_count),
                "rollouts": int(total_rollouts),
                "pass_count": int(row.pass_at_8_count),
                "pass_rate": float(row.pass_at_8_fraction_percent),
                "delta": float(row.pass_at_8_delta_vs_direct_pp),
            }
        )

    baseline = rows[0]["pass_rate"]
    rows.append(
        {
            "label": "Teacher from prefix",
            "kind": "teacher",
            "units": int(teacher["units"]),
            "rollouts": int(teacher["total_rollouts"]),
            "pass_count": int(teacher["pass_at_8_count"]),
            "pass_rate": float(teacher["pass_at_8"] * 100),
            "delta": float((teacher["pass_at_8"] * 100) - baseline),
        }
    )
    return pd.DataFrame(rows)


def main() -> None:
    configure_fonts()
    df = load_data()
    baseline = float(df.iloc[0]["pass_rate"])
    y = list(range(len(df) - 1, -1, -1))

    colors = {"student": "#4B5563", "bridge": "#0072B2", "teacher": "#D55E00"}
    markers = {"student": "s", "bridge": "o", "teacher": "D"}

    fig = plt.figure(figsize=(12.8, 7.1), facecolor="white")
    grid = fig.add_gridspec(1, 2, width_ratios=[3.2, 2.05], wspace=0.04)
    ax = fig.add_subplot(grid[0, 0])
    ax_table = fig.add_subplot(grid[0, 1], sharey=ax)

    for yi, row in zip(y, df.itertuples()):
        color = colors[row.kind]
        ax.plot(
            [baseline, row.pass_rate],
            [yi, yi],
            color=color,
            linewidth=3.0 if row.kind == "bridge" else 2.3,
            alpha=0.72,
            solid_capstyle="round",
            zorder=2,
        )
        ax.scatter(
            row.pass_rate,
            yi,
            s=112 if row.kind == "bridge" else 125,
            marker=markers[row.kind],
            color=color,
            edgecolor="white",
            linewidth=1.2,
            zorder=3,
        )
        ax.text(
            row.pass_rate + 0.25,
            yi,
            f"{row.pass_rate:.2f}%",
            va="center",
            ha="left",
            fontsize=11.5,
            fontweight="bold",
            color=color,
        )

    ax.axvline(baseline, color="#6B7280", linestyle=(0, (4, 4)), linewidth=1.15, zorder=1)
    ax.text(
        baseline + 0.10,
        len(df) - 0.28,
        f"Student 基线 {baseline:.2f}%",
        fontsize=9.5,
        color="#4B5563",
        va="top",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"], fontsize=11)
    ax.set_xlim(67.2, 86.0)
    ax.set_ylim(-0.7, len(df) - 0.15)
    ax.set_xlabel("Pass@8：8 次 rollout 中至少一次正确 (%)", fontsize=11.5, labelpad=10)
    ax.set_xticks([68, 72, 76, 80, 84])
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
    header_y = len(df) - 0.15
    ax_table.text(0.02, header_y, "Pass groups / units", fontsize=10.2,
                  fontweight="bold", color="#374151", va="bottom")
    ax_table.text(0.60, header_y, "Rollouts", fontsize=10.2,
                  fontweight="bold", color="#374151", ha="center", va="bottom")
    ax_table.text(0.98, header_y, "提升", fontsize=10.2,
                  fontweight="bold", color="#374151", ha="right", va="bottom")

    for yi, row in zip(y, df.itertuples()):
        ax_table.text(0.02, yi, f"{row.pass_count:,} / {row.units:,}",
                      va="center", fontsize=10.2, color="#374151")
        ax_table.text(0.60, yi, f"{row.rollouts:,}", va="center", ha="center",
                      fontsize=10.2, color="#374151")
        delta_text = "—" if abs(row.delta) < 1e-10 else f"+{row.delta:.2f} pp"
        ax_table.text(
            0.98,
            yi,
            delta_text,
            va="center",
            ha="right",
            fontsize=10.2,
            fontweight="bold" if row.delta > 0 else "normal",
            color=colors[row.kind] if row.delta > 0 else "#6B7280",
        )
        ax_table.axhline(yi - 0.5, color="#F1F5F9", linewidth=0.8, xmin=0.01, xmax=0.99)

    fig.suptitle(
        "不同 Bridge 支持力度下的 Pass@8",
        x=0.055,
        y=0.97,
        ha="left",
        fontsize=17,
        fontweight="bold",
        color="#111827",
    )
    fig.text(
        0.055,
        0.916,
        "每个统计单位均使用 8 次 rollout；连线表示相对 Student from prefix 的差异。",
        fontsize=10.5,
        color="#4B5563",
        ha="left",
    )
    fig.text(
        0.055,
        0.025,
        "注：bridge 行的单位是 prefix–bridge（13,244 prefixes × 8 bridges），其总 rollout 数更大；通过率的分子、分母同步扩大。",
        fontsize=9.3,
        color="#6B7280",
        ha="left",
    )

    fig.subplots_adjust(left=0.26, right=0.975, top=0.86, bottom=0.14)
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUTPUT_PNG)


if __name__ == "__main__":
    main()
