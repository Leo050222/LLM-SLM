# 项目代码与运行入口

整理日期：2026-10-19。本文记录本地源码、实验入口、可视化入口和路径约束；实验是否完成、实际分数和远程产物位置以实验总表及原始结果为准。

## 目录职责

| 目录 | 内容与使用方式 |
| --- | --- |
| `scripts/` | 数据准备、训练、评测、分数汇总、可视化导出，以及对应 Slurm 启动文件。保留现有物理路径，避免破坏脚本间导入及旧作业记录。 |
| `ocrd_pipeline/anchor_selector/` | 学生/教师 token 分布评分、候选位置生成、外部或本地 judge、锚点选择。 |
| `ocrd_pipeline/bridge_generator/` | 以锚点为输入，生成固定长度教师 bridge，并核验首 token 约束。 |
| `ocrd_pipeline/full_rollout/` | 教师完整回答、学生从 anchor 继续、学生在 bridge 后继续，及多次采样的评分与汇总。 |
| `ocrd_pipeline/online_rollout/` | 分离 GPU worker 与联网 coordinator 的在线接管；共享文件队列支持恢复。 |
| `experiments/<实验名>/` | 各实验的固定配置、启动器、状态和结果摘要。新增研究优先在独立实验目录保留配置与运行来源。 |
| `analysis/` | 已完成分析的复算脚本、CSV/JSON、中英文解释和图。分析脚本与其输入输出暂留同处。 |
| `reports/`、`figures/` | 人读报告、离线 HTML、表格和论文图；其中部分子目录含生成报告的 Python 脚本。 |
| `tools/opd_token_heatmap/` | 离线 token heatmap 的 HTML 模板和格式说明。 |
| `code/bridge-rollout-browser/` | 建议用于发布的 bridge 对比浏览器源码；React + Vite，数据来自 `public/data/`。 |
| 根目录 `bridge-rollout-browser/` | 相同浏览器的独立 Git 工作目录，含本地依赖和构建产物；保留本地历史，不重复发布。 |
| `remote_patch/attention_utils.py` | 外部训练依赖的补丁，属于复现材料，不按临时脚本删除。 |

分类明细见 [code_inventory.csv](code_inventory.csv)；JSON 版本 [code_inventory.json](code_inventory.json) 还记录各 Python 文件静态导入的模块。该清单覆盖 `scripts/`、`ocrd_pipeline/`、`analysis/`、`tools/` 中的 Python、Shell 和 Slurm 文件，未把生成数据、浏览器源码或每个实验目录的脚本混入清单。

## 实验调用关系

```text
数据准备 scripts/build_*、prepare_*、download_*
    ├─ OPD / Reverse-KL 训练 → checkpoint → baseline / checkpoint 评测
    └─ 学生 rollout → token 诊断与 teacher 评分
          → anchor_selector 候选与 judge
          → selected_anchor.json
          → bridge_generator 教师 bridge
          → full_rollout 学生继续 / 教师完整回答
          → correctness / length 评分与 n8 / x8 汇总
          → analysis / reports / figures / browser 数据导出

online_rollout：prompts → GPU workers + 联网 coordinator
    → completed / traces / judge_cache → evaluate.py
```

各阶段保留原始 manifest、seed、采样设置与模型路径。`full_rollout` 的多采样 pass 指标与 `online_rollout` 的单次回答准确率属于不同协议，不能直接合并为同一指标。

### 数据和训练

| 目标 | 主要入口 | 配套启动方式 |
| --- | --- | --- |
| DeepMath difficulty ≥ 6 数据与分层划分 | `scripts/build_deepmath_ge6_opd_data.py`、`build_deepmath_ge6_train15pct.py`、`build_deepmath_ge6_train25pct_perdifficulty.py` | 相同名称的 `submit_build_*.slurm`，并运行相应 `verify_*.py` |
| GSM8K / MATH 数据 | `scripts/build_gsm8k_math_opd_data.py` | 对应 GKD launcher |
| verl Reverse-KL / GKD | `scripts/run_qwen3_1_7b_from_14b_reverse_kl_deepmath_ge6.sh`、`run_qwen3_4b_from_14b_gkd_gsm8k_math.sh` | 对应 `submit_qwen3_*.slurm` |
| 全词表或 student-top-k Reverse-KL | `scripts/run_deepmath_full_vocab_reverse_kl_train.py` | `run_deepmath_top16_*launcher.sh` 和 `submit_deepmath_*rkl*.slurm`；top-k 由环境配置决定 |
| 保存 bridge 的 student-top16 训练 | `scripts/run_bridge_top16_reverse_kl_train.py` | `run_bridge_top16_8h800_single_node_launcher.sh` 与 `submit_*bridge*top16*.slurm` |

`run_bridge_top16_reverse_kl_train.py` 直接导入同目录的 `run_deepmath_full_vocab_reverse_kl_train.py`；两文件必须一起部署。训练入口会在导入时校验配置和数据，不适合在缺少数据的本地环境中盲目执行。

### 生成、评分与诊断

| 目标 | 入口 |
| --- | --- |
| Qwen hard benchmarks | `scripts/run_qwen3_hard_eval.py`、`run_qwen3_hard_eval_nonthinking.py` 及同名 Shell launcher |
| DeepMath rollout / checkpoint x8 | `scripts/run_deepmath_train25_nonthinking_rollout.py`、`run_deepmath_checkpoint_nonthinking_x8.py` |
| 学生、教师 token 诊断 | `scripts/run_opd_logit_diagnostic.py`、`run_deepmath_student_top16_trace_diagnostic.py` |
| 数学结果评分 | `scripts/score_format_free_final_answer.py`、`score_deepmath_checkpoint_nonthinking_x8.py`、`score_deepmath_train25_nonthinking_rollout.py` |
| shard 合并与完整性检查 | `scripts/merge_*.py`、`verify_*.py`，以及 pipeline 内的 `aggregate_*` / `verify_*` |

锚点阶段的具体命令、输入输出格式见 [anchor selector README](../../ocrd_pipeline/anchor_selector/README.md)。典型顺序为 `score-student` → `score-teacher` → `candidates` → `prepare-judge` → `judge-prepared`。计算节点不联网，联网 judge 与 GPU 评分按阶段分开。

Bridge 阶段用 `ocrd_pipeline/bridge_generator/run_bridge_generator.py --anchor ... --output ... --h 32`；批量入口为同目录 `run_bridge_batch.py`。具体固定长度与首 token 限制见 [bridge README](../../ocrd_pipeline/bridge_generator/README.md)。

在线接管依次使用 `ocrd_pipeline/online_rollout/prepare.py`、`launch_workers*.slurm`、`run.py` / `run_random.py`、`evaluate.py`。详细环境变量及恢复约定见 [online README](../../ocrd_pipeline/online_rollout/README.md)。

## 可视化与复算

| 产物 | 生成代码 | 输入 |
| --- | --- | --- |
| 离线 token KL heatmap | `scripts/export_opd_token_heatmap.py` | `tokens.parquet` 或包含该文件的诊断目录 |
| baseline gap 页面 | `scripts/build_deepmath_baseline_gap_vis.py`、`serve_deepmath_baseline_gap.py` | baseline 样本及 token 分析；模板在 `tools/opd_token_heatmap/` |
| student-top16 对比页 | `scripts/build_deepmath_student_top16_trace_vis.py` | 对应 trace diagnostic / merge 输出 |
| bridge 浏览器数据 | `scripts/export_bridge_100pp_browser_data.py` | 集群已保存的 direct / after-bridge traces 和 x8 scores；向 stdout 输出 gzip tar |
| 论文图与分布图 | `analysis/plot_*.py`、`reports/**/plot_*.py` | 同名或相邻 CSV/JSON，以及脚本内记录的原始数据路径 |
| 历史分析复算 | `analysis/*_202609*.py`、`analysis/research_direction_review_20260911/recompute.py` | 脚本内声明的数据集、汇总与路径 |

离线 heatmap 示例（在仓库根执行，显式指定输出）：

```bash
python scripts/export_opd_token_heatmap.py /path/to/diagnostic/tokens.parquet \
  --output reports/example_token_heatmap.html --title "Token KL diagnostic"
```

浏览器使用 Node ≥ 22.13.0；`package.json` 和 `pnpm-lock.yaml` 固定了当前依赖。其有效入口为 `index.html` → `src/main.tsx` → `app/page.tsx` 和 `app/globals.css`。

```bash
cd code/bridge-rollout-browser
pnpm install --frozen-lockfile
pnpm dev
# 生产构建：pnpm build；构建预览：pnpm preview
```

默认开发地址由 `vite.config.ts` 固定为 `127.0.0.1:4173`。浏览器运行需要相应的 `public/data/`，不能把该目录当作编译缓存直接删除。

## 环境和路径约束

源码静态导入显示，GPU 运行依赖主要包括 PyTorch、Transformers、vLLM、verl，以及部分入口使用的 evalscope、datasets、safetensors；分析依赖 NumPy、pandas、PyArrow、Matplotlib、SciPy。当前项目没有统一且经过验证的根级 Python 锁定环境，使用集群已验证环境和各实验保存的配置。静态导入清单不等于可复现安装锁文件，未推断 CUDA 兼容版本。

按照根目录 [AGENT.md](../../AGENT.md)，公共数据和模型只读：

```text
/storage/yukaichengLab/share/datasets
/storage/yukaichengLab/share/model
```

所有新增 checkpoint、rollout、Slurm 日志、评测、可视化、Hydra 和临时输出都放在 `/storage/yukaichengLab/wangxin/test_eiloeo/<分类目录>`。登录节点仅用于环境准备、下载和提交作业；长时间训练或推理必须通过 Slurm。

旧版 Slurm 大量指向 `/storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/`，而本地目录名为 `scripts/`。部署时需同步到原位置或提供明确的兼容入口，并同步 `tools/opd_token_heatmap/` 中的模板。不能仅把源码上传到一个新的 `OPD/scripts/` 后直接重跑旧 Slurm。

`analysis/` 多个绘图脚本用 `Path(__file__).resolve().parents[1]` 查找仓库根；`recompute.py` 使用 `parents[2]`。`scripts/export_opd_token_heatmap.py` 也依赖自身到根目录的层级。因此本次以分类入口和索引规整代码、分析、报告、图像，保留物理路径。仅保留旧路径的符号链接无法修复 `resolve()` 后的层级变化。

## 清理和发布审计

机器清单见 [cleanup_candidates.json](cleanup_candidates.json)。该文件记录候选及依据，本代码审计本身没有删除或移动现有文件；实际执行以项目整理记录为准。

可以清理 Python bytecode、Matplotlib / pnpm 缓存、`node_modules/`、`tsconfig.tsbuildinfo` 和 `.DS_Store`。浏览器两套目录中各自的 `vite.config 2.ts`、`app/page 2.tsx`、`app/globals 2.css` 是未被有效入口引用的旧 Sites/Next 模板，可清理；它们与无后缀文件并非逐字相同。

浏览器两套目录排除 `.git`、`node_modules`、`dist`、`public` 后，共 81 个文件逐文件完全相同。建议只发布 `code/bridge-rollout-browser/` 这一份源码。未据此判定两套 `public/data` 或构建目录全部相同，不整目录盲删。根目录版本自己的 `.git` 属于独立历史，应保留。

`tmp/` 含历史 judge 结果、checkpoint baseline 摘要、候选 bundle 和启动日志；其中不全是可重建缓存，适合归档为历史实验产物。`remote_patch/` 也保留。未完成内容相等与引用核查的 `reports/**/data_legacy_json/* 2.js`、`EXPERIMENT_TRACKER 2.md` 不列入安全删除范围。

代码与启动器的凭据字面量扫描完成，命中 `ocrd_pipeline/anchor_selector/.secrets.env:1` 和 `.secrets.openlux.env:1`；两文件均被 Git ignore 且未受版本控制。凭据值不写入文档或扫描输出。根 `.gitignore` 还明确标记三个历史报告文件包含第三方 API key material，发布需继续排除这些文件：

- `reports/deepmath_student_top16_trace_difference.html`
- `reports/deepmath_baseline_sample_gap/data_legacy_json/case_0004.js`
- `reports/deepmath_baseline_sample_gap/data_legacy_json/case_0069.js`

这次扫描覆盖本代码清单及本地 secret env；不代表所有实验日志、数据文件和既往 Git 历史均已经完成安全审计。推送前应检查最终暂存文件列表及其凭据扫描结果。
