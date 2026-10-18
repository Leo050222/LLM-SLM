# GitHub 发布检查

检查日期：2026-10-19（Asia/Shanghai）。目标仓库：[Leo050222/LLM-SLM](https://github.com/Leo050222/LLM-SLM)。本报告不记录任何凭据值。

## 仓库状态

- 原工作目录：`/Users/leolee/Documents/OPD`。
- 原 Git 分支为 `master`，只有一个提交：`0393b2a`（`Add OPD code, rollouts, and analysis artifacts`）。
- 原 `origin` 指向 `https://github.com/Leo050222/LLMSLMCol.git`，应保留。
- 对目标执行 `git ls-remote --symref ... HEAD 'refs/heads/*'` 成功，返回为空：检查时目标仓库没有 HEAD 或分支。它与本地旧提交尚无可比较的远程提交关系。
- `gh` 未安装；Git 使用 `osxkeychain` 凭据助手。此项仅说明认证方式，读取远程 refs 不证明具有写入权限；最终以 push 结果为准。
- 原仓库已跟踪 54,001 个文件，当前这些文件合计约 1.47 GB；`.git` 中的 loose objects 约 1.07 GiB。原历史包含逐样本结果、报告嵌入数据和凭据材料，不宜直接推送。

## 已确认需要排除的文件

以下行号针对整理前的原文件；只报告位置，不报告凭据内容。

| 路径 | 位置 | 原因 |
| --- | --- | --- |
| `ocrd_pipeline/anchor_selector/.secrets.env` | 第 1 行 | 匹配 API token |
| `ocrd_pipeline/anchor_selector/.secrets.openlux.env` | 第 1 行 | 匹配 API token |
| `reports/deepmath_student_top16_trace_difference.html` | 第 24 行 | 嵌入内容含 Google API key 前缀 |
| `reports/deepmath_baseline_sample_gap/data_legacy_json/case_0004.js` | 第 1 行 | 嵌入内容含 Google API key 前缀 |
| `reports/deepmath_baseline_sample_gap/data_legacy_json/case_0069.js` | 第 1 行 | 嵌入内容含 Google API key 前缀 |

后三项已经存在于旧提交中，即使写入 `.gitignore` 也不会从旧历史移除。通过新建干净发布快照避免带入旧历史。密钥有效性没有进行外部试用；如曾真实使用，应由密钥所有者轮换。

所有 `.env`、`.env.*`、`.secrets*`、私钥文件、本地账号配置和代理/会话状态均不应进入发布快照。还需排除 `.git/`、嵌套仓库元数据、`node_modules/`、缓存、日志、临时目录、数据集及逐样本原始输出。

发现嵌套仓库：`bridge-rollout-browser/.git`。保留所需源文件时排除其 Git 元数据，避免把源码误提交为缺少子模块配置的 gitlink。

## 大文件

旧提交中三个超过 50 MiB 的文件：

| 路径 | 字节数 |
| --- | ---: |
| `reports/deepmath_baseline_sample_gap.tar` | 97,396,560 |
| `reports/crux_length_20260916/data.js` | 72,804,914 |
| `reports/deepmath_student_top16_trace_difference.html` | 55,190,842 |

工作目录另有超过 100 MiB 的存储检查输出和临时归档，均应留在共享存储/本地数据区：

- `outputs/h800-bos-inspection/wangxin-full-search-20261011/auwomo-data-processed-matches.jsonl`
- `outputs/h800-bos-inspection/wangxin-full-search-20261011/wangxin-matches.csv`
- `outputs/h800-bos-inspection/wangxin-full-search-20261011/auwomo-data-processed-wangxin-matches.csv`
- `tmp/anchor_train25_qwen3_1_7b_nonthinking_seed20260811_transformers_v2/candidate_records_and_manifests.tar.zst`

GitHub 对大于 50 MiB 的普通 Git 文件给出警告，阻止大于 100 MiB 的文件；仓库应保持精简。见 [GitHub 大文件说明](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)。本项目发布代码、配置、复现脚本、文档与紧凑结果表；大型数据和运行产物通过共享存储路径与清单定位。

## 发布方式及验收

1. 在独立发布目录生成允许清单控制的快照，保留原仓库及其历史。
2. 复制实际代码、可视化源码、配置、必要启动脚本、项目文档与紧凑汇总；不要复制原 `.git`。
3. 对快照重新执行凭据模式检查、禁止文件类型检查、最大文件大小检查、嵌套 `.git` 检查和文档链接检查。
4. 在发布目录初始化独立 Git 仓库，提交前检查 `git diff --cached --stat` 与文件清单。
5. 推送到用户指定目标，随后用远程 refs 校验提交哈希。目标目前为空，无需覆盖原有远程历史。

检查局限：静态凭据模式检查不能证明内容绝无敏感信息，也未验证任何 token 的权限或有效性。完整发布快照的最终检查与 push 证据应以整理任务的交付记录为准。
