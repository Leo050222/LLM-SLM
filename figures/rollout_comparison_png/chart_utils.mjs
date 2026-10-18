import fs from "node:fs/promises";
import path from "node:path";
import sharp from "sharp";

export const DIR = path.dirname(new URL(import.meta.url).pathname);
export const COLORS = ["#0072B2", "#009E73", "#E69F00"];

export async function loadData() {
  return JSON.parse(await fs.readFile(path.join(DIR, "rollout_metrics.json"), "utf8")).experiments;
}

export const esc = (s) => String(s).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
export const pct = (x) => `${(x * 100).toFixed(2)}%`;
export const comma = (x) => Number(x).toLocaleString("en-US");

export function multilineText(x, y, lines, attrs = "") {
  const spans = lines.map((line, i) => `<tspan x="${x}" dy="${i === 0 ? 0 : 28}">${esc(line)}</tspan>`).join("");
  return `<text x="${x}" y="${y}" ${attrs}>${spans}</text>`;
}

export function axes({ width, height, margin, maxY, yLabel, tickCount = 5 }) {
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;
  const out = [];
  for (let i = 0; i <= tickCount; i++) {
    const value = maxY * i / tickCount;
    const y = margin.top + plotH - plotH * i / tickCount;
    out.push(`<line x1="${margin.left}" y1="${y}" x2="${margin.left + plotW}" y2="${y}" stroke="#D9DEE5" stroke-width="1"/>`);
    out.push(`<text x="${margin.left - 18}" y="${y + 7}" text-anchor="end" class="tick">${Math.round(value * 100)}%</text>`);
  }
  out.push(`<line x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${margin.top + plotH}" stroke="#4A5563" stroke-width="2"/>`);
  out.push(`<line x1="${margin.left}" y1="${margin.top + plotH}" x2="${margin.left + plotW}" y2="${margin.top + plotH}" stroke="#4A5563" stroke-width="2"/>`);
  out.push(`<text transform="translate(34 ${margin.top + plotH / 2}) rotate(-90)" text-anchor="middle" class="axis-label">${esc(yLabel)}</text>`);
  return { markup: out.join(""), plotW, plotH };
}

export function wrapSvg(width, height, content) {
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
  <rect width="100%" height="100%" fill="#FFFFFF"/>
  <style>
    text { font-family: Arial, "PingFang SC", sans-serif; fill: #1F2933; }
    .tick { font-size: 24px; }
    .axis-label { font-size: 28px; font-weight: 500; }
    .category { font-size: 26px; }
    .value { font-size: 25px; font-weight: 500; }
    .legend { font-size: 25px; }
    .note { font-size: 22px; fill: #52606D; }
  </style>
  ${content}
  </svg>`;
}

export async function savePng(svg, filename) {
  const out = path.join(DIR, filename);
  await sharp(Buffer.from(svg)).png({ compressionLevel: 9 }).withMetadata({ density: 300 }).toFile(out);
  console.log(out);
}
