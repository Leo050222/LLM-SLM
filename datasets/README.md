# 数据集目录

本目录统一说明数据位置、准备入口和复现约束。完整数据、大型模型与实验载荷保存在西湖共享存储，本仓库保存代码、数据 manifest 和选定汇总。

## 存储位置

| 数据 | 位置 | 使用约定 |
| --- | --- | --- |
| 公共原始数据集 | `/storage/yukaichengLab/share/datasets/` | 只读复用，下载前先检查是否已有 |
| 公共模型 | `/storage/yukaichengLab/share/model/` | 只读复用，不在仓库中保存权重 |
| 项目处理数据 | 西湖项目根目录下 `opd_data/`、`dataset/`、`benchmarks/` | 保留划分、seed、manifest、源哈希和 tokenizer 信息 |
| 本地工作数据 | `datasets/local` → 根目录 `data/` | 相对符号链接；数据文件不默认随 GitHub 快照发布 |
| 已同步数据摘要 | 根目录 `results/westlake/opd_data/` | manifest / 状态等小文件，不是完整数据集副本 |
| 下载工具与记录 | 根目录 `tools/downloads/` | 模型与视觉 benchmark 的下载、校验和准备工具 |

西湖项目根目录为 `/storage/yukaichengLab/wangxin/test_eiloeo`。连接与文件传输见 [WESTLAKE.md](../docs/storage/WESTLAKE.md)：

```bash
python3 tools/storage/westlake.py list opd_data
python3 tools/storage/westlake.py list benchmarks
```

以上命令在仓库根执行。原始回答、rollout、score 和 checkpoint 属于实验产物，不混入共享原始数据目录。

## 数据准备入口

| 数据处理 | 代码 |
| --- | --- |
| DeepMath difficulty ≥ 6 标准 OPD 数据 | [build_deepmath_ge6_opd_data.py](../scripts/build_deepmath_ge6_opd_data.py) |
| DeepMath 15% 训练子集 | [build_deepmath_ge6_train15pct.py](../scripts/build_deepmath_ge6_train15pct.py) |
| DeepMath 按难度分层 25% 子集 | [build_deepmath_ge6_train25pct_perdifficulty.py](../scripts/build_deepmath_ge6_train25pct_perdifficulty.py) |
| GSM8K / MATH | [build_gsm8k_math_opd_data.py](../scripts/build_gsm8k_math_opd_data.py) |
| DeepMath 评测切片 | [prepare_deepmath_ge6_eval_shards.py](../scripts/prepare_deepmath_ge6_eval_shards.py) |
| 在线接管固定 prompts | [online_rollout/prepare.py](../ocrd_pipeline/online_rollout/prepare.py) |

对应验证器和 Slurm 启动器见 [脚本分类索引](../scripts/README.md)。评测输入应固定样本集合、问题 ID、prompt / tokenizer、模型、thinking 设置和随机种子；分片合并时核对总行数、重复和缺失。复现某个结果前从 [实验总表](../docs/experiments/ALL_EXPERIMENTS.md) 找到该实验自己的输入与配置，不用同名 benchmark 的其他切片替代。

新增数据处理结果写入西湖项目根目录内，禁止修改共享原始数据；保持训练集、验证集和测试集划分可追溯。

## 已保存的数据指纹

[manifest.csv](manifest.csv) 从西湖已保存的 DeepMath 三份顶层 manifest 提取文件路径、划分行数、SHA256、seed 与训练 padding 行数。只填源文件明确提供的字段，空白表示该源未提供；同一测试集在多个训练方案中重复使用，不计作不同测试集。源 manifest 保存在 `results/westlake/opd_data/`。
