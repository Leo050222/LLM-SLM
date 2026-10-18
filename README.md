# LLM-SLM

大模型与小模型协同推理、教师 bridge、在线接管和 on-policy distillation 的实验代码、结果索引与研究文档。项目包含 DeepMath 训练与评测、数学和代码 benchmark、不同接管策略及介入机制分析。

## 从这里开始

| 需要做什么 | 入口 |
| --- | --- |
| 查全部实验、对应结果和证据状态 | [实验与结果总目录](docs/experiments/ALL_EXPERIMENTS.md) |
| 找实验、训练、评分和可视化代码 | [代码地图](docs/project/CODE_MAP.md)、[脚本分类索引](scripts/README.md) |
| 了解整理后的目录与兼容入口 | [项目目录说明](docs/project/DIRECTORY_LAYOUT.md) |
| 连接西湖共享存储，查找原始数据 | [西湖存储说明](docs/storage/WESTLAKE.md) |
| 查数据、源码和可视化约定 | [数据集](datasets/README.md)、[源码](src/README.md)、[可视化](visualization/README.md) |

实验总目录区分已保存的结果、仅有部署记录的实验和仍在计划中的分析，并提供逐系列表格、机器清单及源文件定位。比较结果时需保持数据集、模型、thinking 设置、采样次数与评分协议一致；准确率、pass@8 和 all@8 分别记录。

## 目录概览

```text
docs/                       实验总表、研究计划、审计、论文与存储说明
scripts/                    通用数据准备、训练、评分、验证及 Slurm 脚本
ocrd_pipeline/              anchor → teacher bridge → continuation / online 流程
experiments/                按实验保存的代码、配置、启动记录和结果摘要
src/                        源码分类入口、外部依赖补丁、西湖代码快照
visualization/              浏览器、分析、报告和图表的统一入口
analysis/ reports/ figures/ 历史分析、报告和图像的有效物理路径
results/                    结果分类入口、汇总表和西湖小结果快照
datasets/                   数据存放与复用约定，本地数据入口
tools/                      存储访问、目录整理、下载和 HTML 模板工具
artifacts/                  历史临时产物归档、原始输出入口、存储盘点
```

2026-10-19 的整理保留了代码和分析脚本依赖的目录层级，用相对符号链接提供分类入口。研究文档、下载工具、历史临时文件和补丁已归入对应目录；旧路径保留兼容链接。逐项变更与清理记录见 [ORGANIZATION_CHANGES.json](docs/project/ORGANIZATION_CHANGES.json)。

## 西湖存储与运行

旧西湖共享存储已通过现有 SSH 别名 `gpu-wangxin` 验证连接，登录节点为 `login01`，项目根目录为 `/storage/yukaichengLab/wangxin/test_eiloeo`。

```bash
python3 tools/storage/westlake.py status
python3 tools/storage/westlake.py list
python3 tools/storage/westlake.py shell
```

访问需要可达西湖内网及已授权的个人 SSH 密钥，具体配置和 rsync 用法见 [存储说明](docs/storage/WESTLAKE.md)。大型 checkpoint、全量 rollout、数据集和 BF16 logits 保留在西湖；本地 [results/westlake](results/westlake) 保存已同步的小结果汇总和 manifest，[src/westlake_snapshot](src/westlake_snapshot) 保存代码快照。

长时间 GPU 训练和推理遵循 [AGENT.md](AGENT.md)，通过 Slurm 提交。公共模型与数据只读复用，新增输出统一放在西湖项目根目录下。各历史 launcher 仍可能使用集群的 `opd_scripts/` 等绝对路径，部署前按 [代码地图](docs/project/CODE_MAP.md) 核对环境和路径。

浏览器可视化使用 Node ≥ 22.13.0：

```bash
cd code/bridge-rollout-browser
pnpm install --frozen-lockfile
pnpm dev
```

浏览器需要对应 `public/data/`。离线 token heatmap、报告导出与图表复算见 [可视化说明](visualization/README.md)。Python GPU 入口沿用各实验保存的已验证环境；当前没有统一的根级 Python 环境锁文件。

## 仓库内容范围

目标仓库为 [Leo050222/LLM-SLM](https://github.com/Leo050222/LLM-SLM)。发布内容是整理后的代码、文档、配置、图表与选定结果摘要；原始 Git 历史、本地凭据、可重建缓存和大型实验载荷不随快照发布。发布审查记录见 [PUBLICATION_REVIEW.md](docs/project/PUBLICATION_REVIEW.md)。本文件不作为远程推送完成凭证。
