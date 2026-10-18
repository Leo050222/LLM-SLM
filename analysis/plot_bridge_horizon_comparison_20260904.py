#!/usr/bin/env python3
"""Create comparison tables and PNG figures for 32/64/256-token bridges."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties


ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
TAG = "20260906"

FONT_PATH = Path("/System/Library/Fonts/STHeiti Medium.ttc")
CN = FontProperties(fname=str(FONT_PATH)) if FONT_PATH.exists() else None
mpl.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 220,
    "axes.unicode_minus": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.20,
    "grid.linewidth": 0.7,
    "font.size": 10,
})

H = [32, 64, 128, 256]
COLORS = {32: "#3B82F6", 64: "#10B981", 128: "#8B5CF6", 256: "#F59E0B"}
BASE_ACC = 39.26683781334944
SUMMARY = {
    32: dict(acc=40.499471458773784, lift=1.2326336454243414, full_mean=1342.0572228461945,
             full_median=1164, cont_mean=997.6846767876019, cont_median=836,
             full_delta=-6.450988419284204, clean_reduction=6.906921830260535,
             pearson=0.07389307056504094, spearman=0.0688586038827013,
             shorter_lift=2.293392699468335, other_lift=0.16304761180407667,
             length_finish=81, zero_bridge=23.270440251572327,
             zero_prefix=53.94291243347847, zero_acc=5.464592404450895,
             perfect_bridges=157, perfect_prefixes=37, perfect_multi_bridges=152,
             perfect_multi_prefixes=32),
    64: dict(acc=41.001113711869525, lift=1.7342758985200823, full_mean=1341.1702256682272,
             full_median=1164, cont_mean=964.7976796096345, cont_median=804,
             full_delta=-7.337985597251586, clean_reduction=7.89249298343429,
             pearson=0.0937510010708211, spearman=0.08222215293769909,
             shorter_lift=3.119725788405742, other_lift=0.3611927365883062,
             length_finish=84, zero_bridge=23.50024189646831,
             zero_prefix=53.87034349298501, zero_acc=5.8777062167392355,
             perfect_bridges=214, perfect_prefixes=68, perfect_multi_bridges=184,
             perfect_multi_prefixes=38),
    128: dict(acc=42.016667925098156, lift=2.7498301117487134, full_mean=1347.8518397481878,
              full_median=1167, cont_mean=907.4792936895952, cont_median=741,
              full_delta=-0.6563715172908486, clean_reduction=6.51004035993119,
              pearson=0.11509292316498484, spearman=0.10279949289932444,
              shorter_lift=4.7195410319029345, other_lift=0.8827761788797874,
              length_finish=190, zero_bridge=24.377116594097725,
              zero_prefix=55.53942912433478, zero_acc=6.76176221577165,
              perfect_bridges=344, perfect_prefixes=131, perfect_multi_bridges=278,
              perfect_multi_prefixes=65),
    256: dict(acc=44.20645669737239, lift=4.9396188840229505, full_mean=1400.1579524218514,
              full_median=1177, cont_mean=831.7854063632589, cont_median=619,
              full_delta=51.6497411563727, clean_reduction=-3.14292964424633,
              pearson=0.1343591142783457, spearman=0.1317040157862942,
              shorter_lift=7.870604156987901, other_lift=2.328780539924496,
              length_finish=1083, zero_bridge=26.276003870343495,
              zero_prefix=58.65989356555394, zero_acc=8.795204402515723,
              perfect_bridges=731, perfect_prefixes=275, perfect_multi_bridges=614,
              perfect_multi_prefixes=158),
}

LIFT_HIST = {
    32: [23, 39, 92, 355, 1199, 3270, 7466, 15448, 46661, 16613, 8048, 3832, 1627, 664, 277, 181, 157],
    64: [29, 49, 135, 439, 1249, 3220, 7354, 15099, 46002, 16382, 8333, 4148, 1846, 802, 412, 239, 214],
    128: [57, 79, 204, 532, 1294, 3255, 6923, 14253, 45172, 16219, 8714, 4439, 2284, 1120, 652, 411, 344],
    256: [88, 174, 298, 756, 1434, 3079, 6306, 12485, 43608, 15912, 9066, 5134, 3135, 1744, 1139, 863, 731],
}
LIFT_STEPS = np.arange(-8, 9)
LIFT_PP = LIFT_STEPS * 12.5

CLEAN_REDUCTION_BY_LIFT = {
    32: [-132.6522, -89.0929, -125.5109, -29.3271, -20.0569, -26.3098, -15.4612, -4.7445,
         3.0353, 17.1480, 30.3256, 46.3303, 58.4355, 86.5919, 100.2920, 85.4137, 94.8288],
    64: [-79.1681, -63.4643, -117.6713, -29.0271, -32.9079, -32.8260, -23.5798, -6.7007,
         3.2280, 14.4748, 37.1009, 63.0439, 78.4075, 105.8015, 135.4402, 122.7118, 95.1297],
    128: [-92.6607, -4.0253, -93.8701, -56.0679, -38.2698, -42.3765, -32.6766, -13.7598,
          -2.2173, 13.4904, 40.4501, 75.8055, 95.1992, 113.4200, 148.3257, 166.2848, 105.1582],
    256: [-71.2226, -4.0571, -41.0998, -61.7477, -67.6266, -60.2708, -52.1679, -30.7834,
          -22.5037, 3.0007, 39.1295, 68.1958, 108.9783, 143.6750, 139.0189, 141.9820, 118.3289],
}

ZERO_HIST = {
    32: [25376, 4560, 1607, 700, 313, 167, 103, 89, 157],
    64: [25300, 4358, 1679, 735, 350, 200, 134, 102, 214],
    128: [25010, 4242, 1650, 793, 435, 268, 173, 157, 344],
    256: [24382, 4025, 1655, 829, 514, 364, 280, 292, 731],
}
PERFECT_MULTIPLICITY = {
    32: [5, 7, 7, 1, 4, 4, 3, 6],
    64: [30, 4, 8, 8, 2, 8, 2, 6],
    128: [66, 20, 13, 7, 5, 6, 2, 12],
    256: [117, 54, 32, 27, 10, 10, 8, 17],
}


def zh(text: str, **kwargs):
    kwargs.setdefault("fontproperties", CN)
    return text, kwargs


def title(ax, text):
    ax.set_title(text, fontproperties=CN, loc="left", pad=10)


def annotate_bars(ax, bars, fmt="{:.1f}", dy=3):
    for bar in bars:
        value = bar.get_height()
        ax.annotate(fmt.format(value), (bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, dy), textcoords="offset points", ha="center", va="bottom", fontsize=8)


def write_csvs():
    with (ROOT / "analysis" / f"bridge_horizon_summary_{TAG}.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["horizon_tokens"] + list(next(iter(SUMMARY.values())).keys())
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for horizon in H:
            writer.writerow({"horizon_tokens": horizon, **SUMMARY[horizon]})
    with (ROOT / "analysis" / f"bridge_lift_bucket_trace_{TAG}.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["horizon_tokens", "lift_steps_of_8", "lift_pp", "bridge_count", "clean_mean_whole_trace_reduction_tokens"])
        for horizon in H:
            for step, pp, count, reduction in zip(LIFT_STEPS, LIFT_PP, LIFT_HIST[horizon], CLEAN_REDUCTION_BY_LIFT[horizon]):
                writer.writerow([horizon, step, pp, count, reduction])
    with (ROOT / "analysis" / f"zero_baseline_recovery_gradient_{TAG}.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["horizon_tokens", "correct_rollouts_after_bridge_out_of_8", "bridge_count", "bridge_fraction_percent"])
        for horizon in H:
            total = sum(ZERO_HIST[horizon])
            for correct, count in enumerate(ZERO_HIST[horizon]):
                writer.writerow([horizon, correct, count, 100 * count / total])


def overview():
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    fig.suptitle("Bridge 长度对正确率与 trace length 的影响", fontproperties=CN, fontsize=16)

    ax = axes[0, 0]
    bars = ax.bar([str(h) for h in H], [SUMMARY[h]["lift"] for h in H], color=[COLORS[h] for h in H], width=.62)
    annotate_bars(ax, bars, "+{:.2f} pp")
    ax.axhline(0, color="#555555", linewidth=.8)
    ax.set_ylabel("正确率提升（百分点）", fontproperties=CN)
    ax.set_xlabel("Bridge token 数", fontproperties=CN)
    title(ax, f"A. 正确率提升（原 prefix 基线 {BASE_ACC:.2f}%）")

    ax = axes[0, 1]
    x = np.arange(len(H) + 1)
    labels = ["原 prefix"] + [str(h) for h in H]
    full = [1348.5082] + [SUMMARY[h]["full_mean"] for h in H]
    cont = [1036.1357] + [SUMMARY[h]["cont_mean"] for h in H]
    ax.plot(x, full, marker="o", linewidth=2, label="整体 trace", color="#7C3AED")
    ax.plot(x, cont, marker="s", linewidth=2, label="Student continuation", color="#0891B2")
    for xx, yy in zip(x, full):
        ax.annotate(f"{yy:.0f}", (xx, yy), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8)
    ax.set_xticks(x, labels)
    for label in ax.get_xticklabels():
        label.set_fontproperties(CN)
    ax.set_ylabel("平均 token 数", fontproperties=CN)
    ax.set_xlabel("条件", fontproperties=CN)
    ax.legend(prop=CN, frameon=False)
    title(ax, "B. 平均 trace length")

    ax = axes[1, 0]
    width = .35
    xx = np.arange(len(H))
    b1 = ax.bar(xx - width/2, [SUMMARY[h]["shorter_lift"] for h in H], width, label="trace 变短", color="#2563EB")
    b2 = ax.bar(xx + width/2, [SUMMARY[h]["other_lift"] for h in H], width, label="未变短", color="#94A3B8")
    annotate_bars(ax, b1, "{:.2f}", 2)
    annotate_bars(ax, b2, "{:.2f}", 2)
    ax.set_xticks(xx, [str(h) for h in H])
    ax.set_ylabel("平均正确率提升（百分点）", fontproperties=CN)
    ax.set_xlabel("Bridge token 数", fontproperties=CN)
    ax.legend(prop=CN, frameon=False)
    title(ax, "C. Bridge 级：trace 变短组提升更大")

    ax = axes[1, 1]
    width = .25
    b1 = ax.bar(xx - width, [SUMMARY[h]["zero_bridge"] for h in H], width, label="bridge 有提升", color="#0EA5E9")
    b2 = ax.bar(xx, [SUMMARY[h]["zero_prefix"] for h in H], width, label="prefix 至少一个 bridge 提升", color="#10B981")
    b3 = ax.bar(xx + width, [SUMMARY[h]["zero_acc"] for h in H], width, label="续写 pooled 正确率", color="#F59E0B")
    ax.set_xticks(xx, [str(h) for h in H])
    ax.set_ylabel("占比 / 正确率（%）", fontproperties=CN)
    ax.set_xlabel("Bridge token 数", fontproperties=CN)
    ax.legend(prop=CN, frameon=False, fontsize=8)
    title(ax, "D. 原 prefix 0/8（共 4,134 条）的恢复")

    path = FIG / f"bridge_horizon_overview_{TAG}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def lift_distribution():
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    for horizon in H:
        fraction = np.asarray(LIFT_HIST[horizon]) / sum(LIFT_HIST[horizon]) * 100
        ax.plot(LIFT_PP, fraction, marker="o", markersize=4, linewidth=2,
                color=COLORS[horizon], label=f"{horizon}-token")
    ax.axvline(0, color="#555555", linewidth=.8)
    ax.set_xticks(LIFT_PP, [f"{v:+.1f}" if v else "0" for v in LIFT_PP], rotation=45, ha="right")
    ax.set_xlabel("正确率提升档位（百分点；8 次 rollout）", fontproperties=CN)
    ax.set_ylabel("全部 bridge 的占比（%）", fontproperties=CN)
    ax.legend(frameon=False)
    title(ax, "全部 bridge 的正确率提升分布")
    path = FIG / f"bridge_horizon_lift_distribution_{TAG}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def trace_correlation():
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True, constrained_layout=True)
    fig.suptitle("不同正确率提升档位中的整体 trace length 变化", fontproperties=CN, fontsize=15)
    for ax, horizon in zip(axes, H):
        y = np.asarray(CLEAN_REDUCTION_BY_LIFT[horizon])
        sizes = 15 + 50 * np.sqrt(np.asarray(LIFT_HIST[horizon]) / max(LIFT_HIST[horizon]))
        ax.plot(LIFT_PP, y, color=COLORS[horizon], linewidth=1.8)
        ax.scatter(LIFT_PP, y, s=sizes, color=COLORS[horizon], alpha=.85)
        ax.axhline(0, color="#555555", linewidth=.8)
        ax.axvline(0, color="#555555", linewidth=.8)
        ax.set_xticks([-100, -50, 0, 50, 100])
        ax.set_xlabel("正确率提升（百分点）", fontproperties=CN)
        title(ax, f"{horizon}-token：Pearson r={SUMMARY[horizon]['pearson']:.3f}")
    axes[0].set_ylabel("平均整体 trace 缩短 token 数\n（正值=变短；排除 length finish）", fontproperties=CN)
    path = FIG / f"bridge_horizon_trace_correlation_{TAG}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def perfect_concentration():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    fig.suptitle("+100 pp bridge 的 prefix 集中度", fontproperties=CN, fontsize=15)
    ax = axes[0]
    xx = np.arange(len(H))
    width = .34
    bridge_share = [100 * SUMMARY[h]["perfect_multi_bridges"] / SUMMARY[h]["perfect_bridges"] for h in H]
    prefix_share = [100 * SUMMARY[h]["perfect_multi_prefixes"] / SUMMARY[h]["perfect_prefixes"] for h in H]
    b1 = ax.bar(xx-width/2, bridge_share, width, label="来自重复出现 prefix 的 +100 pp bridge", color="#8B5CF6")
    b2 = ax.bar(xx+width/2, prefix_share, width, label="含 ≥2 个 +100 pp bridge 的 prefix", color="#22C55E")
    annotate_bars(ax, b1, "{:.1f}%")
    annotate_bars(ax, b2, "{:.1f}%")
    ax.set_ylim(0, 108)
    ax.set_xticks(xx, [str(h) for h in H])
    ax.set_xlabel("Bridge token 数", fontproperties=CN)
    ax.set_ylabel("占比（%）", fontproperties=CN)
    ax.legend(prop=CN, frameon=False, fontsize=8)
    title(ax, "A. 同一 prefix 重复贡献的占比")

    ax = axes[1]
    mult = np.arange(1, 9)
    for horizon in H:
        vals = np.asarray(PERFECT_MULTIPLICITY[horizon])
        ax.plot(mult, vals / vals.sum() * 100, marker="o", linewidth=2,
                color=COLORS[horizon], label=f"{horizon}-token (n={vals.sum()} prefixes)")
    ax.set_xticks(mult)
    ax.set_xlabel("同一 prefix 中 +100 pp bridge 数", fontproperties=CN)
    ax.set_ylabel("有 +100 pp 的 prefix 占比（%）", fontproperties=CN)
    ax.legend(frameon=False, fontsize=8)
    title(ax, "B. 每个 prefix 的 +100 pp bridge 数分布")
    path = FIG / f"bridge_horizon_plus100_concentration_{TAG}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    write_csvs()
    overview()
    lift_distribution()
    trace_correlation()
    perfect_concentration()
    print("wrote 3 CSV tables and 4 PNG figures")
