# 视觉 benchmark 离线数据

已下载并通过文件大小、固定内容哈希、结构读取和图片抽样验证。

远端目录：`/storage/yukaichengLab/wangxin/test_eiloeo/benchmarks/vision`

共 8 个 benchmark，58 个源文件，3,289,506,357 字节（约 3.29 GB）。

| Benchmark | 当前目录内容 | 数据大小 |
|---|---|---:|
| [MathVista](https://huggingface.co/datasets/AI4Math/MathVista) | `data/test-00000-of-00002-6b81bd7f7e2065e6.parquet`：2,571 条<br>`data/test-00001-of-00002-6a611c71596db30f.parquet`：2,570 条<br>`data/testmini-00000-of-00001-725687bf7a18d64b.parquet`：1,000 条 | 885.8 MB |
| [MathVision](https://huggingface.co/datasets/MathLLMs/MathVision) | `data/test-00000-of-00001-3532b8d3f1b4047a.parquet`：3,040 条<br>`data/testmini-00000-of-00001-f8ff70fcb2f29b1d.parquet`：304 条 | 63.9 MB |
| [MathVerse](https://huggingface.co/datasets/AI4Math/MathVerse) | `testmini.parquet`：3,940 条<br>`testmini_text_only.parquet`：788 条 | 167.2 MB |
| [CV-Bench](https://huggingface.co/datasets/nyu-visionx/CV-Bench) | `test_2d.parquet`：1,438 条<br>`test_3d.parquet`：1,200 条 | 404.8 MB |
| [BLINK](https://huggingface.co/datasets/BLINK-Benchmark/BLINK) | 全部 14 个任务；val 1,901 条，test 1,906 条 | 797.3 MB |
| [MMBench](https://github.com/open-compass/MMBench) | `MMBench_DEV_EN_V11.tsv`：4,876 条<br>`MMBench_DEV_CN_V11.tsv`：4,876 条<br>`MMBench_TEST_EN_V11.tsv`：7,299 条<br>`MMBench_TEST_CN_V11.tsv`：7,299 条 | 250.3 MB |
| [MMStar](https://huggingface.co/datasets/Lin-Chen/MMStar) | `mmstar.parquet`：1,500 条 | 41.8 MB |
| [RealWorldQA](https://huggingface.co/datasets/xai-org/RealworldQA) | `data/test-00000-of-00002.parquet`：383 条<br>`data/test-00001-of-00002.parquet`：382 条 | 678.3 MB |

## 版本与评测范围

- MathVista：testmini 1,000 题可本地计分；test 5,141 题的答案由官方保留。
- MathVision：test 为完整 3,040 题；testmini 304 题是其子集，不要相加作为独立题数。
- MathVerse：保存官方全部公开内容，含 788 基题的 5 种视觉版本共 3,940 条，以及 788 条 Text Only；论文更大的完整版没有公开发布。
- CV-Bench：完整 2D 与 3D 两部分。
- BLINK：全部 14 个任务的 val/test；val 答案公开，test 答案隐藏。以本地 Parquet 实际行数为准。
- MMBench：v1.1，英文／中文的 DEV／TEST 共四份；表中行数含 CircularEval 选项轮换，不等于原始题数。下面列出实际非空答案数量。图片列包含 base64 和行号引用，读取时需解析。
- MMStar：完整 val，1,500 题。RealWorldQA：完整 test，765 题，答案公开。

  - `MMBench_DEV_EN_V11.tsv`：4,876/4,876 条有非空答案；4,876 条图片可解析。
  - `MMBench_DEV_CN_V11.tsv`：4,876/4,876 条有非空答案；4,876 条图片可解析。
  - `MMBench_TEST_EN_V11.tsv`：7,299/7,299 条有非空答案；7,299 条图片可解析。
  - `MMBench_TEST_CN_V11.tsv`：7,299/7,299 条有非空答案；7,299 条图片可解析。

MMBench 两份 TEST 使用当前官方公开答案版本，其内容哈希与 VLMEvalKit 代码内登记的 MD5 存在差异。已通过独立下载计算 SHA-256，再与 H800 实存文件交叉复核；登记值、实际值与来源均记录在 `MMBench/SOURCE.json`。

Parquet 内嵌图片字节，可离线读取，不需要再下载同一批原始图片压缩包。

## 离线读取

集群现有环境可直接使用：

```bash
PY=/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python
ROOT=/storage/yukaichengLab/wangxin/test_eiloeo/benchmarks/vision
$PY "$ROOT/load_benchmark.py" --root "$ROOT" --list
$PY "$ROOT/load_benchmark.py" --root "$ROOT" --benchmark MathVista --split testmini --limit 2
$PY "$ROOT/load_benchmark.py" --root "$ROOT" --benchmark BLINK --task Relative_Depth --split val --limit 2
$PY "$ROOT/load_benchmark.py" --root "$ROOT" --benchmark MMBench --split DEV --task EN --limit 2
```

Python 中可使用 `load_benchmark.iter_benchmark(root, benchmark, split=..., task=...)`。

## 来源与完整性

`plan.json` 固定官方仓库 revision、文件清单和预期大小／哈希；各数据目录的 `SOURCE.json` 与 `SHA256SUMS` 记录实存文件。
`VERIFIED.json` 记录每份数据的行数、图片和答案统计及校验结果；`status.json` 与 `download.log` 记录下载状态。
各 Hugging Face 仓库的原始 README 和可用许可证文件一并保存，使用条件以原始数据说明为准。

如需重新验证：

```bash
$PY "$ROOT/verify.py" --root "$ROOT"
```
