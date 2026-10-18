# 源码分类入口

本目录集中导航核心 pipeline、实验代码、依赖补丁和西湖代码快照。通用脚本继续保存在根目录 `scripts/`，按 [脚本分类索引](../scripts/README.md) 查找。

| 入口 | 内容 |
| --- | --- |
| `pipeline` → 根目录 `ocrd_pipeline/` | anchor selection、teacher bridge、continuation 评测与 online intervention |
| `experiments` → 根目录 `experiments/` | 按实验命名的配置、实现、Slurm、部署与评分记录 |
| `patches/` | 原 `remote_patch/` 的外部训练依赖补丁 |
| `westlake_snapshot/` | 已同步的西湖 `opd_scripts/` 与 `opd_online_code/` 代码快照，保留原目录结构 |

`pipeline`、`experiments` 为相对符号链接，不复制源码。原目录保持物理位置，脚本内的同目录导入和 `Path(__file__).resolve()` 层级继续有效。

核心阶段说明：

- [Anchor selector](../ocrd_pipeline/anchor_selector/README.md)：分开执行学生 / 教师评分、候选生成、judge 和锚点选择。
- [Teacher bridge](../ocrd_pipeline/bridge_generator/README.md)：固定长度 bridge 与首 token 限制。
- [Online rollout](../ocrd_pipeline/online_rollout/README.md)：GPU worker、联网 coordinator、共享文件队列与恢复约定。
- [完整代码地图](../docs/project/CODE_MAP.md)：训练、评分、可视化、依赖与集群路径。

西湖快照用于保留远端实现与本地未收录的历史版本，不自动替换本地同名文件，也不表示所有快照入口均在当前环境重新验证。复现前从 [实验总表](../docs/experiments/ALL_EXPERIMENTS.md) 核对实际配置、模型、评分协议和结果来源。

GPU 入口沿用实验自己的已验证环境，并遵循 [AGENT.md](../AGENT.md)；旧 Slurm 可能指向西湖的 `opd_scripts/`，部署时保持相应路径或明确配置兼容入口。共享存储访问见 [WESTLAKE.md](../docs/storage/WESTLAKE.md)。
