# 项目目录说明

整理日期：2026-10-19。目录以代码、实验、可视化、文档、结果、原始产物和数据的职责组织。已有实验入口依赖的物理路径保留，统一分类目录通过相对符号链接提供导航。

主要入口：[实验与结果总表](../experiments/ALL_EXPERIMENTS.md)、[代码地图](CODE_MAP.md)、[西湖共享存储](../storage/WESTLAKE.md)。完整机器变更记录为 [ORGANIZATION_CHANGES.json](ORGANIZATION_CHANGES.json)。

## 分类入口

| 分类目录或入口 | 实际内容 / 目标 | 说明 |
| --- | --- | --- |
| `src/pipeline` | `ocrd_pipeline/` | anchor、teacher bridge、continuation 和 online rollout |
| `src/experiments` | `experiments/` | 每个实验的代码、配置与启动记录 |
| `src/patches/` | 原 `remote_patch/` | 外部依赖补丁，保留复现来源 |
| `src/westlake_snapshot/` | 西湖 `opd_scripts/`、`opd_online_code/` 快照 | 远端历史代码与启动器，保留原相对结构 |
| `scripts/` | 原目录 | 通用脚本按 [分类索引](../../scripts/README.md) 查找，继续同级部署 |
| `visualization/browser` | `code/bridge-rollout-browser/` | 用于发布的 React + Vite 浏览器源码 |
| `visualization/analysis` | `analysis/` | 分析和绘图脚本及相邻结果 |
| `visualization/reports` | `reports/` | HTML、分析文档和报告构建脚本 |
| `visualization/figures` | `figures/` | 图表和图表源数据 |
| `results/reports`、`results/figures`、`results/analysis` | 同名根目录 | 从结果视角浏览已有报告、图和分析 |
| `results/tables/` | 项目级汇总表 | 包括 greedy triplet 表 |
| `results/westlake/` | 已同步的远端小结果和 manifest | 大载荷仍在西湖 |
| `datasets/local` | `data/` | 本地数据入口；数据不默认纳入 GitHub 快照 |
| `artifacts/outputs` | `outputs/` | 原始输出和检查产物的兼容入口 |
| `artifacts/legacy_tmp/` | 原 `tmp/` | 历史 judge、baseline、候选包与启动记录 |
| `artifacts/storage_inventory/` | 存储盘点材料 | 用于追溯文件位置与迁移状态 |
| `tools/downloads/` | 原 `downloads/` | 模型 / benchmark 下载工具及记录 |

分类目录中的符号链接不产生第二份数据，也不改变原始文件路径。分析、报告、图像目录同时从 `visualization/` 和 `results/` 可见，分别方便找绘图入口与查结果。

## 已移动的十项

以下源路径均保留相对符号链接，旧代码和文档仍可访问：

| 旧路径 | 新物理位置 |
| --- | --- |
| `tmp/` | `artifacts/legacy_tmp/` |
| `downloads/` | `tools/downloads/` |
| `refine-logs/` | `docs/research/refine-logs/` |
| `idea-stage/` | `docs/research/idea-stage/` |
| `papers/` | `docs/papers/` |
| `MANIFEST.md` | `docs/research/MANIFEST.md` |
| `EXPERIMENT_AUDIT.md` | `docs/audits/EXPERIMENT_AUDIT.md` |
| `EXPERIMENT_AUDIT.json` | `docs/audits/EXPERIMENT_AUDIT.json` |
| `analysis_current_greedy_triplets.csv` | `results/tables/analysis_current_greedy_triplets.csv` |
| `remote_patch/` | `src/patches/` |

`scripts/`、`ocrd_pipeline/`、`experiments/`、`analysis/`、`reports/`、`figures/` 保持物理位置。部分脚本用 `Path(__file__).resolve().parents[...]` 找仓库根，另有同目录 Python 导入；随意移动会改变行为，仅添加旧路径链接也无法修复 `resolve()` 后的层级。

## 数据和结果的边界

代码、配置、计划、汇总结果和原始输出分开理解：

- `docs/experiments/` 是所有实验的证据索引，明确结果状态与单位。
- `results/`、`reports/`、`figures/` 和 `analysis/` 保存汇总、图和可复算材料。
- `results/westlake/` 是远端小结果快照，不等于所有远端大文件的本地副本。
- 西湖共享存储保留完整 dataset、checkpoint、rollout、logits 和大归档；具体位置见 [WESTLAKE.md](../storage/WESTLAKE.md)。

GPU 任务的新增输出必须写入 `/storage/yukaichengLab/wangxin/test_eiloeo/<分类目录>`；公共模型和数据目录只读。旧西湖根目录下的 `organized/` 提供分类入口，原始实验目录继续有效。

## 清理和 GitHub 快照

本次清理包含 Python / Node / pnpm / Matplotlib / TypeScript 可重建缓存、macOS 目录元数据、浏览器构建产物及未使用的旧模板。逐项记录见 [ORGANIZATION_CHANGES.json](ORGANIZATION_CHANGES.json)，清理判断依据见 [cleanup_candidates.json](cleanup_candidates.json)。历史临时目录中的实验证据已归档，未按目录名一概删除。

本地根目录的 `bridge-rollout-browser/` 保留自己的 Git 历史；GitHub 快照只选用 `code/bridge-rollout-browser/` 这一份源码。快照排除个人凭据、原始历史、缓存和大型载荷；本地数据入口或历史兼容路径可能仅在完整工作目录内可用。文档主要链接指向发布的代码与文档物理路径，便于在 GitHub 浏览。
