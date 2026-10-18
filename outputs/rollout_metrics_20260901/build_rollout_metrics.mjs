import fs from "node:fs/promises";
import { Workbook, SpreadsheetFile } from "@oai/artifact-tool";

const outputDir = "/Users/leolee/Documents/OPD/outputs/rollout_metrics_20260901";
const outputPath = `${outputDir}/rollout_experiment_metrics.xlsx`;
const previewPath = `${outputDir}/dashboard_preview.png`;
const rawPreviewPath = `${outputDir}/raw_summary_preview.png`;
const distributionPreviewPath = `${outputDir}/distribution_preview.png`;

const wb = Workbook.create();
const dashboard = wb.worksheets.add("Dashboard");
const raw = wb.worksheets.add("Raw Summary");
const dist = wb.worksheets.add("Distribution");

const navy = "#17324D";
const blue = "#3B82F6";
const green = "#10B981";
const orange = "#F59E0B";
const paleBlue = "#EAF2FF";
const paleGreen = "#E8F7F1";
const paleOrange = "#FFF4DA";
const light = "#F5F7FA";
const border = "#D7DEE8";
const muted = "#5F6B7A";

for (const sheet of [dashboard, raw, dist]) sheet.showGridLines = false;

// Raw experiment-level inputs and formula-derived metrics.
raw.getRange("A1:N1").values = [[
  "Experiment",
  "Metric unit",
  "Prefixes",
  "Units evaluated",
  "Bridges / prefix",
  "Rollouts / unit",
  "Total rollouts",
  "Correct rollouts",
  "Accuracy",
  "Pass@8 units",
  "Pass@8",
  "All 8 correct units",
  "All 8 correct rate",
  "Strict summary source",
]];

raw.getRange("A2:H4").values = [
  ["Student from prefix", "prefix", 13244, 13244, 0, 8, 105952, 41604],
  ["Student after bridge", "bridge", 13244, 105952, 8, 8, 847616, 374701],
  ["Teacher from prefix", "prefix", 13244, 13244, 0, 8, 105952, 60723],
];
raw.getRange("J2:J4").values = [[9110], [74060], [11007]];
raw.getRange("L2:L4").values = [[1560], [20702], [3599]];
raw.getRange("N2:N4").values = [
  ["student_from_anchor_prefix.../analysis_correctness_length_single_job/summary.json"],
  ["student_after_bridge...x8.../aggregate/summary.json"],
  ["teacher_full.../analysis_correctness_length_single_job/summary.json"],
];
raw.getRange("I2").formulas = [["=H2/G2"]];
raw.getRange("I2:I4").fillDown();
raw.getRange("K2").formulas = [["=J2/D2"]];
raw.getRange("K2:K4").fillDown();
raw.getRange("M2").formulas = [["=L2/D2"]];
raw.getRange("M2:M4").fillDown();

raw.getRange("A1:N1").format = {
  fill: navy,
  font: { bold: true, color: "#FFFFFF" },
  wrapText: true,
  verticalAlignment: "center",
  borders: { preset: "outside", style: "thin", color: navy },
};
raw.getRange("A2:N4").format.borders = {
  insideHorizontal: { style: "thin", color: border },
  bottom: { style: "thin", color: border },
};
raw.getRange("I2:I4").format.numberFormat = "0.00%";
raw.getRange("K2:K4").format.numberFormat = "0.00%";
raw.getRange("M2:M4").format.numberFormat = "0.00%";
raw.getRange("C2:H4").format.numberFormat = "#,##0";
raw.getRange("J2:J4").format.numberFormat = "#,##0";
raw.getRange("L2:L4").format.numberFormat = "#,##0";
raw.getRange("A1:N4").format.rowHeight = 24;
raw.getRange("A:A").format.columnWidth = 24;
raw.getRange("B:B").format.columnWidth = 13;
raw.getRange("C:M").format.columnWidth = 14;
raw.getRange("N:N").format.columnWidth = 64;
raw.freezePanes.freezeRows(1);
raw.tables.add("A1:N4", true, "RolloutSummaryTable");

// Normalized 0..8 correctness distributions.
dist.getRange("A1:G1").values = [[
  "Correct among 8",
  "Student count",
  "Student rate",
  "Bridge count",
  "Bridge rate",
  "Teacher count",
  "Teacher rate",
]];
const studentHist = [4134, 1449, 1095, 986, 915, 928, 1012, 1165, 1560];
const bridgeHist = [31892, 10183, 7533, 6624, 6314, 6454, 7312, 8938, 20702];
const teacherHist = [2237, 1090, 876, 821, 930, 913, 1105, 1673, 3599];
dist.getRange("A2:A10").values = Array.from({ length: 9 }, (_, i) => [i]);
dist.getRange("B2:B10").values = studentHist.map((x) => [x]);
dist.getRange("D2:D10").values = bridgeHist.map((x) => [x]);
dist.getRange("F2:F10").values = teacherHist.map((x) => [x]);
dist.getRange("C2").formulas = [["=B2/'Raw Summary'!$D$2"]];
dist.getRange("C2:C10").fillDown();
dist.getRange("E2").formulas = [["=D2/'Raw Summary'!$D$3"]];
dist.getRange("E2:E10").fillDown();
dist.getRange("G2").formulas = [["=F2/'Raw Summary'!$D$4"]];
dist.getRange("G2:G10").fillDown();
dist.getRange("A1:G1").format = {
  fill: navy,
  font: { bold: true, color: "#FFFFFF" },
  wrapText: true,
};
dist.getRange("A2:G10").format.borders = {
  insideHorizontal: { style: "thin", color: border },
};
dist.getRange("B2:B10").format.numberFormat = "#,##0";
dist.getRange("D2:D10").format.numberFormat = "#,##0";
dist.getRange("F2:F10").format.numberFormat = "#,##0";
dist.getRange("C2:C10").format.numberFormat = "0.00%";
dist.getRange("E2:E10").format.numberFormat = "0.00%";
dist.getRange("G2:G10").format.numberFormat = "0.00%";
dist.getRange("A:G").format.columnWidth = 18;
dist.freezePanes.freezeRows(1);
dist.tables.add("A1:G10", true, "CorrectnessDistributionTable");

// Dashboard title and KPI cards.
dashboard.mergeCells("A1:N2");
dashboard.getRange("A1").values = [["Rollout Experiment Comparison"]];
dashboard.getRange("A1:N2").format = {
  fill: navy,
  font: { bold: true, color: "#FFFFFF", size: 20 },
  horizontalAlignment: "left",
  verticalAlignment: "center",
};
dashboard.mergeCells("A3:N3");
dashboard.getRange("A3").values = [[
  "13,244 anchor prefixes · accuracy uses every rollout · Direct/Teacher pass@8 unit = prefix · Bridge pass@8 unit = bridge",
]];
dashboard.getRange("A3:N3").format = {
  fill: light,
  font: { color: muted, italic: true, size: 10 },
  wrapText: true,
};

const cards = [
  ["A5:D5", "A6:D7", "Student accuracy", "='Raw Summary'!I2", paleBlue, blue],
  ["F5:I5", "F6:I7", "Bridge accuracy", "='Raw Summary'!I3", paleGreen, green],
  ["K5:N5", "K6:N7", "Teacher accuracy", "='Raw Summary'!I4", paleOrange, orange],
];
for (const [labelRange, valueRange, label, formula, fill, accent] of cards) {
  dashboard.mergeCells(labelRange);
  dashboard.mergeCells(valueRange);
  dashboard.getRange(labelRange.split(":")[0]).values = [[label]];
  dashboard.getRange(valueRange.split(":")[0]).formulas = [[formula]];
  dashboard.getRange(labelRange).format = {
    fill,
    font: { bold: true, color: accent, size: 11 },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    borders: { preset: "outside", style: "thin", color: accent },
  };
  dashboard.getRange(valueRange).format = {
    fill,
    font: { bold: true, color: navy, size: 22 },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    numberFormat: "0.00%",
    borders: { preset: "outside", style: "thin", color: accent },
  };
}

// Formula-backed summary table that drives charts.
dashboard.getRange("A10:H10").values = [[
  "Experiment",
  "Metric unit",
  "Accuracy",
  "Pass@8",
  "All 8 correct rate",
  "All 8 correct count",
  "Units evaluated",
  "Total rollouts",
]];
dashboard.getRange("A11:A13").formulas = [["='Raw Summary'!A2"], ["='Raw Summary'!A3"], ["='Raw Summary'!A4"]];
dashboard.getRange("B11:B13").formulas = [["='Raw Summary'!B2"], ["='Raw Summary'!B3"], ["='Raw Summary'!B4"]];
dashboard.getRange("C11:C13").formulas = [["='Raw Summary'!I2"], ["='Raw Summary'!I3"], ["='Raw Summary'!I4"]];
dashboard.getRange("D11:D13").formulas = [["='Raw Summary'!K2"], ["='Raw Summary'!K3"], ["='Raw Summary'!K4"]];
dashboard.getRange("E11:E13").formulas = [["='Raw Summary'!M2"], ["='Raw Summary'!M3"], ["='Raw Summary'!M4"]];
dashboard.getRange("F11:F13").formulas = [["='Raw Summary'!L2"], ["='Raw Summary'!L3"], ["='Raw Summary'!L4"]];
dashboard.getRange("G11:G13").formulas = [["='Raw Summary'!D2"], ["='Raw Summary'!D3"], ["='Raw Summary'!D4"]];
dashboard.getRange("H11:H13").formulas = [["='Raw Summary'!G2"], ["='Raw Summary'!G3"], ["='Raw Summary'!G4"]];
dashboard.getRange("A10:H10").format = {
  fill: navy,
  font: { bold: true, color: "#FFFFFF" },
  wrapText: true,
};
dashboard.getRange("A11:H13").format.borders = {
  insideHorizontal: { style: "thin", color: border },
  bottom: { style: "thin", color: border },
};
dashboard.getRange("C11:E13").format.numberFormat = "0.00%";
dashboard.getRange("F11:H13").format.numberFormat = "#,##0";
dashboard.getRange("A:A").format.columnWidth = 24;
dashboard.getRange("B:B").format.columnWidth = 13;
dashboard.getRange("C:E").format.columnWidth = 17;
dashboard.getRange("F:H").format.columnWidth = 18;
dashboard.getRange("I:N").format.columnWidth = 13;
dashboard.freezePanes.freezeRows(3);

dashboard.mergeCells("A15:N16");
dashboard.getRange("A15").values = [[
  "Reading note: Bridge accuracy is directly comparable as the mean success probability of a random bridge continuation. Its pass@8 and all-8-correct metrics are per bridge, while Direct/Teacher use prefix-level groups.",
]];
dashboard.getRange("A15:N16").format = {
  fill: "#FFFBEA",
  font: { color: "#725A00", size: 10 },
  wrapText: true,
  verticalAlignment: "center",
  borders: { preset: "outside", style: "thin", color: "#E6C55A" },
};

// Helper ranges for normalized charts.
dashboard.getRange("O40:R40").values = [["Experiment", "Accuracy", "Pass@8", "All 8 correct"]];
dashboard.getRange("O41:O43").formulas = [["=A11"], ["=A12"], ["=A13"]];
dashboard.getRange("P41:P43").formulas = [["=C11"], ["=C12"], ["=C13"]];
dashboard.getRange("Q41:Q43").formulas = [["=D11"], ["=D12"], ["=D13"]];
dashboard.getRange("R41:R43").formulas = [["=E11"], ["=E12"], ["=E13"]];
dashboard.getRange("P41:R43").format.numberFormat = "0.00%";

const accuracyChart = dashboard.charts.add("bar", dashboard.getRange("O40:Q43"));
accuracyChart.setPosition("A18", "G34");
accuracyChart.title = "Accuracy and pass@8";
accuracyChart.titleTextStyle.fontSize = 13;
accuracyChart.hasLegend = true;
accuracyChart.yAxis = { numberFormatCode: "0%", min: 0, max: 1 };
if (accuracyChart.series.items[0]) accuracyChart.series.items[0].fill = blue;
if (accuracyChart.series.items[1]) accuracyChart.series.items[1].fill = green;

const allCorrectChart = dashboard.charts.add("bar", {
  chartType: "bar",
  title: "All 8 rollouts correct",
  hasLegend: false,
});
const allSeries = allCorrectChart.series.add("All 8 correct rate");
allSeries.categoryFormula = "'Dashboard'!$O$41:$O$43";
allSeries.formula = "'Dashboard'!$R$41:$R$43";
allSeries.fill = orange;
allCorrectChart.setPosition("H18", "N34");
allCorrectChart.titleTextStyle.fontSize = 13;
allCorrectChart.hasLegend = false;
allCorrectChart.yAxis = { numberFormatCode: "0%", min: 0, max: 0.35 };

// A normalized distribution chart on the dedicated sheet.
dist.getRange("I1:L1").values = [["Correct among 8", "Student", "Bridge", "Teacher"]];
for (let r = 2; r <= 10; r++) {
  dist.getRange(`I${r}`).formulas = [[`=A${r}`]];
  dist.getRange(`J${r}`).formulas = [[`=C${r}`]];
  dist.getRange(`K${r}`).formulas = [[`=E${r}`]];
  dist.getRange(`L${r}`).formulas = [[`=G${r}`]];
}
dist.getRange("J2:L10").format.numberFormat = "0.00%";
const distributionChart = dist.charts.add("line", dist.getRange("I1:L10"));
distributionChart.setPosition("I12", "Q31");
distributionChart.title = "Distribution of correct rollouts within each group of 8";
distributionChart.titleTextStyle.fontSize = 13;
distributionChart.hasLegend = true;
distributionChart.yAxis = { numberFormatCode: "0%", min: 0 };
distributionChart.xAxis = { axisType: "textAxis" };
if (distributionChart.series.items[0]) distributionChart.series.items[0].fill = blue;
if (distributionChart.series.items[1]) distributionChart.series.items[1].fill = green;
if (distributionChart.series.items[2]) distributionChart.series.items[2].fill = orange;

await fs.mkdir(outputDir, { recursive: true });
const tableCheck = await wb.inspect({
  kind: "table",
  range: "Dashboard!A10:H13",
  include: "values,formulas",
  tableMaxRows: 10,
  tableMaxCols: 10,
  maxChars: 6000,
});
console.log(tableCheck.ndjson);

const errorCheck = await wb.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "final formula error scan",
  maxChars: 3000,
});
console.log(errorCheck.ndjson);

const preview = await wb.render({ sheetName: "Dashboard", range: "A1:N34", scale: 1.4, format: "png" });
await fs.writeFile(previewPath, new Uint8Array(await preview.arrayBuffer()));
const rawPreview = await wb.render({ sheetName: "Raw Summary", range: "A1:N4", scale: 1.5, format: "png" });
await fs.writeFile(rawPreviewPath, new Uint8Array(await rawPreview.arrayBuffer()));
const distributionPreview = await wb.render({ sheetName: "Distribution", range: "A1:Q31", scale: 1.2, format: "png" });
await fs.writeFile(distributionPreviewPath, new Uint8Array(await distributionPreview.arrayBuffer()));
const output = await SpreadsheetFile.exportXlsx(wb);
await output.save(outputPath);
console.log(JSON.stringify({ outputPath, previewPath, rawPreviewPath, distributionPreviewPath }));
