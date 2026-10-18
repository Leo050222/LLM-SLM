"""Write a concise offline dataset index after successful validation."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    plan = json.loads((root / "plan.json").read_text())
    report = json.loads((root / "VERIFIED.json").read_text())
    status = json.loads((root / "status.json").read_text())
    if not report.get("ok") or status.get("state") != "COMPLETE":
        raise SystemExit("Downloads and validation must complete before writing the final index")
    verified = {dataset["name"]: dataset for dataset in report["datasets"]}
    total = sum(file["size"] for dataset in plan["datasets"] for file in dataset["files"])
    lines = ["# 视觉 benchmark 离线数据", "", "已下载并通过文件大小、固定内容哈希、结构读取和图片抽样验证。",
             "", f"远端目录：`{root}`", "", f"共 8 个 benchmark，58 个源文件，{total:,} 字节（约 {total / 10**9:.2f} GB）。",
             "", "| Benchmark | 当前目录内容 | 数据大小 |", "|---|---|---:|"]
    for dataset in plan["datasets"]:
        name = dataset["name"]
        data = []
        for file in verified[name]["files"]:
            if file.get("kind") in ("parquet", "mmbench_tsv"):
                data.append(f"`{file['path']}`：{file['rows']:,} 条")
        if name == "BLINK":
            val = sum(f.get("rows", 0) for f in verified[name]["files"] if "/val-" in f["path"])
            test = sum(f.get("rows", 0) for f in verified[name]["files"] if "/test-" in f["path"])
            data = [f"全部 14 个任务；val {val:,} 条，test {test:,} 条"]
        size = sum(file["size"] for file in dataset["files"])
        lines.append(f"| [{name}]({dataset['source']}) | {'<br>'.join(data)} | {size / 10**6:.1f} MB |")
    lines += ["", "## 版本与评测范围", "",
              "- MathVista：testmini 1,000 题可本地计分；test 5,141 题的答案由官方保留。",
              "- MathVision：test 为完整 3,040 题；testmini 304 题是其子集，不要相加作为独立题数。",
              "- MathVerse：保存官方全部公开内容，含 788 基题的 5 种视觉版本共 3,940 条，以及 788 条 Text Only；论文更大的完整版没有公开发布。",
              "- CV-Bench：完整 2D 与 3D 两部分。",
              "- BLINK：全部 14 个任务的 val/test；val 答案公开，test 答案隐藏。以本地 Parquet 实际行数为准。",
              "- MMBench：v1.1，英文／中文的 DEV／TEST 共四份；表中行数含 CircularEval 选项轮换，不等于原始题数。下面列出实际非空答案数量。图片列包含 base64 和行号引用，读取时需解析。",
              "- MMStar：完整 val，1,500 题。RealWorldQA：完整 test，765 题，答案公开。", ""]
    for file in verified["MMBench"]["files"]:
        if file.get("kind") == "mmbench_tsv":
            lines.append(f"  - `{file['path']}`：{file['answers_nonempty']:,}/{file['rows']:,} 条有非空答案；{file['images_resolved']:,} 条图片可解析。")
    lines += ["", "MMBench 两份 TEST 使用当前官方公开答案版本，其内容哈希与 VLMEvalKit 代码内登记的 MD5 存在差异。已通过独立下载计算 SHA-256，再与 H800 实存文件交叉复核；登记值、实际值与来源均记录在 `MMBench/SOURCE.json`。",
              "", "Parquet 内嵌图片字节，可离线读取，不需要再下载同一批原始图片压缩包。",
              "", "## 离线读取", "", "集群现有环境可直接使用：", "", "```bash",
              "PY=/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python",
              f"ROOT={root}",
              '$PY "$ROOT/load_benchmark.py" --root "$ROOT" --list',
              '$PY "$ROOT/load_benchmark.py" --root "$ROOT" --benchmark MathVista --split testmini --limit 2',
              '$PY "$ROOT/load_benchmark.py" --root "$ROOT" --benchmark BLINK --task Relative_Depth --split val --limit 2',
              '$PY "$ROOT/load_benchmark.py" --root "$ROOT" --benchmark MMBench --split DEV --task EN --limit 2', "```",
              "", "Python 中可使用 `load_benchmark.iter_benchmark(root, benchmark, split=..., task=...)`。",
              "", "## 来源与完整性", "",
              "`plan.json` 固定官方仓库 revision、文件清单和预期大小／哈希；各数据目录的 `SOURCE.json` 与 `SHA256SUMS` 记录实存文件。",
              "`VERIFIED.json` 记录每份数据的行数、图片和答案统计及校验结果；`status.json` 与 `download.log` 记录下载状态。",
              "各 Hugging Face 仓库的原始 README 和可用许可证文件一并保存，使用条件以原始数据说明为准。",
              "", "如需重新验证：", "", "```bash", '$PY "$ROOT/verify.py" --root "$ROOT"', "```", ""]
    (root / "README.md").write_text("\n".join(lines))
    print("INDEX_COMPLETE", root / "README.md")


if __name__ == "__main__":
    main()
