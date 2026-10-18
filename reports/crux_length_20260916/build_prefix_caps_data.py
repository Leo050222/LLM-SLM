import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "prefix_caps_analysis_20260917"
analysis = json.loads((SOURCE / "analysis.json").read_text())
positions = json.loads((SOURCE / "position_distribution.json").read_text())
samples = []
for line in (SOURCE / "prompts.jsonl").read_text().splitlines():
    row = json.loads(line)
    samples.append(
        {
            "datasetIndex": row["dataset_index"],
            "dataset": row["dataset"],
            "problemId": row["problem_id"],
            "problem": row.get("problem", ""),
            "solution": row.get("solution", ""),
            "subset": row.get("subset", ""),
            "prefixLength": len(row["anchor"]["response_prefix_token_ids"]),
        }
    )
payload = {
    "meta": {
        "title": "固定 prefix 后的 teacher 介入次数实验",
        "threshold": 2,
        "k": 1,
        "samples_per_dataset": 100,
        "rollouts_per_sample": 8,
        "caps": list(range(11)),
        "scoring": "数学与推理数据集使用 GPT-4o-mini；代码数据集使用原有执行评分器",
    },
    "analysis": analysis,
    "positions": positions,
    "samples": samples,
}
(HERE / "prefix_caps_data.js").write_text(
    "window.PREFIX_CAPS_DATA=" + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + ";\n"
)
print(HERE / "prefix_caps_data.js")
