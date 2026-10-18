import { axes, COLORS, comma, loadData, multilineText, pct, savePng, wrapSvg } from "./chart_utils.mjs";

const data = await loadData();
const width = 1650, height = 1050;
const margin = { left: 150, right: 70, top: 120, bottom: 220 };
const { markup, plotW, plotH } = axes({ width, height, margin, maxY: 0.35, yLabel: "All 8 rollouts correct" });
const groupW = plotW / data.length;
const barW = 260;
let body = markup;

data.forEach((d, i) => {
  const center = margin.left + groupW * (i + 0.5);
  const h = plotH * d.all_8_correct_rate / 0.35;
  const y = margin.top + plotH - h;
  body += `<rect x="${center - barW / 2}" y="${y}" width="${barW}" height="${h}" fill="${COLORS[i]}"/>`;
  body += `<text x="${center}" y="${y - 45}" text-anchor="middle" class="value">${pct(d.all_8_correct_rate)}</text>`;
  body += `<text x="${center}" y="${y - 14}" text-anchor="middle" class="note">${comma(d.all_8_correct_count)} / ${comma(d.units)}</text>`;
  body += multilineText(center, margin.top + plotH + 48, d.name.split("\n"), 'text-anchor="middle" class="category"');
});

body += `<text x="${width / 2}" y="${height - 30}" text-anchor="middle" class="note">Grouping unit: prefix for Student/Teacher; bridge for Student after bridge.</text>`;
await savePng(wrapSvg(width, height, body), "rollout_all8_correct.png");
