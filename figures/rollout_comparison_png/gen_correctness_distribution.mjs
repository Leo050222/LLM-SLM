import { axes, COLORS, loadData, savePng, wrapSvg } from "./chart_utils.mjs";

const data = await loadData();
const width = 1800, height = 1080;
const margin = { left: 150, right: 70, top: 170, bottom: 150 };
const { markup, plotW, plotH } = axes({ width, height, margin, maxY: 0.35, yLabel: "Share of groups" });
let body = markup;

function marker(shape, x, y, color) {
  if (shape === "square") {
    return `<rect x="${x - 7}" y="${y - 7}" width="14" height="14" fill="#FFFFFF" stroke="${color}" stroke-width="4"/>`;
  }
  if (shape === "triangle") {
    return `<polygon points="${x},${y - 9} ${x - 9},${y + 8} ${x + 9},${y + 8}" fill="#FFFFFF" stroke="${color}" stroke-width="4" stroke-linejoin="round"/>`;
  }
  return `<circle cx="${x}" cy="${y}" r="7" fill="#FFFFFF" stroke="${color}" stroke-width="4"/>`;
}

for (let k = 0; k <= 8; k++) {
  const x = margin.left + plotW * k / 8;
  body += `<line x1="${x}" y1="${margin.top + plotH}" x2="${x}" y2="${margin.top + plotH + 10}" stroke="#4A5563" stroke-width="2"/>`;
  body += `<text x="${x}" y="${margin.top + plotH + 45}" text-anchor="middle" class="tick">${k}</text>`;
}
body += `<text x="${margin.left + plotW / 2}" y="${height - 40}" text-anchor="middle" class="axis-label">Number correct among 8 rollouts</text>`;

data.forEach((d, i) => {
  const shapes = ["circle", "square", "triangle"];
  const dashes = ["none", "14 8", "4 8"];
  const rates = d.histogram.map((v) => v / d.units);
  const points = rates.map((v, k) => {
    const x = margin.left + plotW * k / 8;
    const y = margin.top + plotH - plotH * v / 0.35;
    return [x, y];
  });
  body += `<polyline points="${points.map((p) => p.join(",")).join(" ")}" fill="none" stroke="${COLORS[i]}" stroke-width="5" stroke-dasharray="${dashes[i]}" stroke-linejoin="round" stroke-linecap="round"/>`;
  points.forEach(([x, y]) => { body += marker(shapes[i], x, y, COLORS[i]); });
  const lx = 380 + i * 480;
  body += `<line x1="${lx}" y1="68" x2="${lx + 55}" y2="68" stroke="${COLORS[i]}" stroke-width="6" stroke-dasharray="${dashes[i]}"/>`;
  body += marker(shapes[i], lx + 27, 68, COLORS[i]);
  body += `<text x="${lx + 72}" y="72" class="legend">${d.name.replace("\n", " ")}</text>`;
  body += `<text x="${lx + 72}" y="102" class="note">unit: ${d.unit}; n=${d.units.toLocaleString("en-US")}</text>`;
});

await savePng(wrapSvg(width, height, body), "rollout_correctness_distribution.png");
