#!/usr/bin/env python3
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
LOGS = {
    93637: ROOT / "opd_dm25_nt_rkl.93637.log",
    93862: ROOT / "opd_dm25_nt_rkl.93862.log",
    94033: ROOT / "opd_dm25_nt_rkl.94033.log",
}


def parse_steps(path: Path):
    rows = []
    with path.open(errors="replace") as handle:
        for line in handle:
            if not line.startswith("DEEPMATH_RKL_STEP_COMPLETE "):
                continue
            payload = json.loads(line.split(" ", 1)[1])
            student = payload["student"]
            rollout = payload["rollout"]
            rank_metrics = student["rank_metrics"]
            rows.append(
                {
                    "step": int(payload["step"]),
                    "reverse_kl": float(student["mean_reverse_kl"]),
                    "grad_norm": float(student["gradient_norm_before_clip"]),
                    "response_tokens": int(student["global_response_tokens"]),
                    "mean_response_tokens": float(student["global_response_tokens"]) / 64.0,
                    "max_response_tokens": int(rollout["max_response_tokens_observed"]),
                    "teacher_seconds": float(payload["teacher_elapsed_max_seconds"]),
                    "student_seconds": max(float(x["elapsed_seconds"]) for x in rank_metrics),
                    "student_peak_allocated_gib": max(float(x["cuda"]["peak_allocated_gib"]) for x in rank_metrics),
                    "student_peak_reserved_gib": max(float(x["cuda"]["peak_reserved_gib"]) for x in rank_metrics),
                    "teacher_peak_allocated_gib": float(payload["teacher_peak_allocated_max_gib"]),
                    "checksum_delta": float(student["parameter_checksum_delta"]),
                    "thinking_tags": int(rollout["generated_thinking_tag_count"]),
                    "stop_count": int(rollout["finish_reasons"].get("stop", 0)),
                }
            )
    return rows


parsed = {job: parse_steps(path) for job, path in LOGS.items()}
main = {r["step"]: r for r in parsed[93637] if r["step"] <= 133}
main.update({r["step"]: r for r in parsed[94033] if r["step"] >= 134})
rows = [main[s] for s in sorted(main)]
if [r["step"] for r in rows] != list(range(1, 223)):
    raise RuntimeError("accepted training curve is not exactly steps 1..222")

fields = list(rows[0])
with (ROOT / "training_curve.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)


def rolling(values, window=10):
    values = np.asarray(values, dtype=float)
    out = np.full(values.shape, np.nan)
    for i in range(window - 1, len(values)):
        out[i] = values[i - window + 1 : i + 1].mean()
    return out


steps = np.array([r["step"] for r in rows])
kl = np.array([r["reverse_kl"] for r in rows])
grad = np.array([r["grad_norm"] for r in rows])
mean_len = np.array([r["mean_response_tokens"] for r in rows])
max_len = np.array([r["max_response_tokens"] for r in rows])
student_sec = np.array([r["student_seconds"] for r in rows])
teacher_sec = np.array([r["teacher_seconds"] for r in rows])

def svg_polyline(xs, ys, box, color, width=1.5, opacity=1.0):
    x0, y0, w, h = box
    finite = np.isfinite(ys)
    xvals, yvals = xs[finite], ys[finite]
    ymin, ymax = float(np.min(yvals)), float(np.max(yvals))
    pad = max((ymax - ymin) * 0.08, 1e-9)
    ymin, ymax = ymin - pad, ymax + pad
    points = []
    for x, y in zip(xvals, yvals):
        px = x0 + (float(x) - 1.0) / 221.0 * w
        py = y0 + h - (float(y) - ymin) / (ymax - ymin) * h
        points.append(f"{px:.1f},{py:.1f}")
    return (
        f'<polyline fill="none" stroke="{color}" stroke-width="{width}" '
        f'opacity="{opacity}" points="{" ".join(points)}"/>', ymin, ymax
    )


panels = [
    ("Restricted reverse KL", kl, rolling(kl), "#7aa6c2", "#0b5a7a", "KL(student || teacher)"),
    ("Gradient norm before clipping", grad, rolling(grad), "#d9a66f", "#a5521d", "global norm"),
    ("Student rollout length", mean_len, rolling(mean_len), "#79a987", "#27633b", "mean response tokens"),
    ("Per-step teacher time", teacher_sec, rolling(teacher_sec), "#d98979", "#963927", "seconds"),
]
svg = [
    '<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="900" viewBox="0 0 1400 900">',
    '<rect width="1400" height="900" fill="#fbfbfa"/>',
    '<style>text{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;fill:#24313a}.title{font-size:24px;font-weight:600}.panel{font-size:18px;font-weight:600}.axis{font-size:12px;fill:#59656d}.note{font-size:12px;fill:#68747c}</style>',
    '<text x="700" y="36" text-anchor="middle" class="title">DeepMath train-25% — non-thinking student Top-16 reverse KL</text>',
]
positions = [(75, 95), (735, 95), (75, 485), (735, 485)]
for (title, raw, smooth, raw_color, smooth_color, ylabel), (px, py) in zip(panels, positions):
    box = (px + 55, py + 45, 560, 275)
    x0, y0, w, h = box
    svg.append(f'<text x="{px}" y="{py+10}" class="panel">{title}</text>')
    svg.append(f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="#ffffff" stroke="#d9dfe2"/>')
    raw_line, ymin, ymax = svg_polyline(steps, raw, box, raw_color, 1.0, 0.42)
    smooth_line, _, _ = svg_polyline(steps, smooth, box, smooth_color, 2.5, 1.0)
    for checkpoint in (44, 89, 133, 178, 222):
        cx = x0 + (checkpoint - 1) / 221 * w
        svg.append(f'<line x1="{cx:.1f}" y1="{y0}" x2="{cx:.1f}" y2="{y0+h}" stroke="#7e878c" stroke-width="0.7" opacity="0.35"/>')
        svg.append(f'<text x="{cx:.1f}" y="{y0+h+18}" text-anchor="middle" class="axis">{checkpoint}</text>')
    svg.extend([raw_line, smooth_line])
    svg.append(f'<text x="{x0-8}" y="{y0+12}" text-anchor="end" class="axis">{ymax:.3g}</text>')
    svg.append(f'<text x="{x0-8}" y="{y0+h}" text-anchor="end" class="axis">{ymin:.3g}</text>')
    svg.append(f'<text x="{x0+w/2}" y="{y0+h+35}" text-anchor="middle" class="axis">optimizer step</text>')
    svg.append(f'<text x="{x0}" y="{y0-10}" class="axis">{ylabel}</text>')
    svg.append(f'<line x1="{x0+w-145}" y1="{y0-10}" x2="{x0+w-115}" y2="{y0-10}" stroke="{raw_color}" opacity="0.5"/><text x="{x0+w-110}" y="{y0-6}" class="note">raw</text>')
    svg.append(f'<line x1="{x0+w-65}" y1="{y0-10}" x2="{x0+w-35}" y2="{y0-10}" stroke="{smooth_color}" stroke-width="2.5"/><text x="{x0+w-30}" y="{y0-6}" class="note">MA10</text>')
svg.append('</svg>')
(ROOT / "training_curves.svg").write_text("\n".join(svg) + "\n")


summary = {
    "steps": len(rows),
    "kl": {
        "first": float(kl[0]),
        "last": float(kl[-1]),
        "mean": float(kl.mean()),
        "std": float(kl.std(ddof=1)),
        "min": float(kl.min()),
        "max": float(kl.max()),
        "first_20_mean": float(kl[:20].mean()),
        "last_20_mean": float(kl[-20:].mean()),
    },
    "grad_norm": {
        "mean": float(grad.mean()),
        "std": float(grad.std(ddof=1)),
        "min": float(grad.min()),
        "max": float(grad.max()),
        "last": float(grad[-1]),
    },
    "response_tokens": {
        "per_step_mean": float(np.mean([r["response_tokens"] for r in rows])),
        "per_step_std": float(np.std([r["response_tokens"] for r in rows], ddof=1)),
        "per_sample_mean": float(mean_len.mean()),
        "max_observed": int(max_len.max()),
    },
    "timing_seconds": {
        "student_mean": float(student_sec.mean()),
        "teacher_mean": float(teacher_sec.mean()),
        "student_p95": float(np.quantile(student_sec, 0.95)),
        "teacher_p95": float(np.quantile(teacher_sec, 0.95)),
    },
    "memory_gib": {
        "student_peak_allocated_max": max(r["student_peak_allocated_gib"] for r in rows),
        "student_peak_reserved_max": max(r["student_peak_reserved_gib"] for r in rows),
        "teacher_peak_allocated_max": max(r["teacher_peak_allocated_gib"] for r in rows),
    },
    "integrity": {
        "total_thinking_tags": sum(r["thinking_tags"] for r in rows),
        "all_batches_64_stops": all(r["stop_count"] == 64 for r in rows),
        "all_checksum_deltas_nonzero": all(r["checksum_delta"] != 0 for r in rows),
    },
    "duplicate_step_134": {
        str(job): next((r for r in values if r["step"] == 134), None)
        for job, values in parsed.items()
        if any(r["step"] == 134 for r in values)
    },
}
(ROOT / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
