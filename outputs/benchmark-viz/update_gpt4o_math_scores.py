#!/usr/bin/env python3
"""Replace math/reasoning accuracy cells in source.md using GPT-4o-mini scores."""

import csv
from pathlib import Path


OUT = Path(__file__).parent
SCORES = OUT.parents[1] / "reports/baseline_math_rescore_20260916/grid_with_baselines.csv"
name_map = {"math500": "MATH500", "olympiad_bench": "OlympiadBench", "bbh": "BBH"}
scores = {}
with SCORES.open() as handle:
    for row in csv.DictReader(handle):
        scores[(row["config"], name_map[row["dataset"]])] = float(row["accuracy_percent"])

lines = (OUT / "source.md").read_text().splitlines()
header = next(line for line in lines if line.startswith("| 实验"))
benchmarks = [cell.strip() for cell in header.split("|")[4:-1]]
updated = []
seen = 0
for line in lines:
    if not line.startswith("|") or line.startswith("| 实验") or line.startswith("|---"):
        updated.append(line)
        continue
    cells = [cell.strip() for cell in line.split("|")[1:-1]]
    if len(cells) != 3 + len(benchmarks):
        updated.append(line)
        continue
    config = cells[0]
    for index, benchmark in enumerate(benchmarks):
        value = scores.get((config, benchmark))
        if value is None:
            continue
        parts = [part.strip() for part in cells[3 + index].split("/")]
        parts[0] = f"{value:.2f}%"
        cells[3 + index] = " / ".join(parts)
        seen += 1
    updated.append("| " + " | ".join(cells) + " |")
if seen != 81:
    raise SystemExit(f"expected 81 updated accuracy cells, got {seen}")
(OUT / "source.md").write_text("\n".join(updated) + "\n")
print(f"updated {seen} accuracy cells")
