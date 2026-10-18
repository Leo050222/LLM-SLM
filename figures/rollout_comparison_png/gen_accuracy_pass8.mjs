import { axes, COLORS, loadData, multilineText, pct, savePng, wrapSvg } from "./chart_utils.mjs";

const data = await loadData();
const width = 1800, height = 1050;
const margin = { left: 150, right: 70, top: 150, bottom: 210 };
const { markup, plotW, plotH } = axes({ width, height, margin, maxY: 0.9, yLabel: "Success rate" });
const groupW = plotW / data.length;
const barW = 150;
let body = markup;

data.forEach((d, i) => {
  const center = margin.left + groupW * (i + 0.5);
  const vals = [d.accuracy, d.pass_at_8];
  vals.forEach((v, j) => {
    const x = center + (j === 0 ? -barW - 12 : 12);
    const h = plotH * v / 0.9;
    const y = margin.top + plotH - h;
    body += `<rect x="${x}" y="${y}" width="${barW}" height="${h}" fill="${COLORS[j]}"/>`;
    if (j === 0) {
      body += `<text x="${x + barW / 2}" y="${y - 18}" text-anchor="middle" class="value">${pct(v)}</text>`;
    } else {
      body += `<text x="${x + barW / 2}" y="${y - 42}" text-anchor="middle" class="value">${pct(v)}</text>`;
      body += `<text x="${x + barW / 2}" y="${y - 12}" text-anchor="middle" class="note">unit: ${d.unit}</text>`;
    }
  });
  body += multilineText(center, margin.top + plotH + 48, d.name.split("\n"), 'text-anchor="middle" class="category"');
});

body += `<rect x="610" y="58" width="28" height="28" fill="${COLORS[0]}"/><text x="653" y="81" class="legend">Accuracy</text>`;
body += `<rect x="890" y="58" width="28" height="28" fill="${COLORS[1]}"/><text x="933" y="81" class="legend">Pass@8</text>`;
body += `<text x="${width / 2}" y="${height - 30}" text-anchor="middle" class="note">Accuracy is rollout-level; each Pass@8 bar states its grouping unit.</text>`;

await savePng(wrapSvg(width, height, body), "rollout_accuracy_pass8.png");
