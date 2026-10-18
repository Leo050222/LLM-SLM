# 西湖共享存储

2026-10-19 按用户提供的 `gpu-wangxin` SSH 配置验证成功，登录节点为 `login01`。本项目的远端根目录是 `/storage/yukaichengLab/wangxin/test_eiloeo`，文件系统挂载点为 `/storage`。

```sshconfig
Host gpu-wangxin
    HostName 172.16.78.35
    Port 10024
    User wangxin
    IdentityFile ~/.ssh/id_ed25519
    PubkeyAuthentication yes
```

需要可访问西湖内网的网络环境，以及管理员已授权的个人 SSH 密钥。密钥不放进仓库。本仓库沿用已有 SSH 配置，不替换本机配置。

```bash
python3 tools/storage/westlake.py status
python3 tools/storage/westlake.py list
python3 tools/storage/westlake.py shell
python3 tools/storage/westlake.py pull organized/results/ ./westlake-results/
python3 tools/storage/westlake.py push README.md organized/docs/README.md
```

`pull` 和 `push` 使用 rsync，远端路径相对项目根目录，不使用 `--delete`。这是 SSH 文件访问入口，并非本机 FUSE 磁盘挂载。

| 内容 | 西湖原始位置 |
|---|---|
| 实验与提交脚本 | `opd_scripts/`、`ocrd_pipeline/`、`opd_online_code/` |
| 训练输出 | `opd_runs/` |
| 评测与消融 | `opd_eval*/`、`opd_rescore/` |
| 分析与诊断 | `opd_analysis/`、`opd_logit_diagnostics/`、`analysis/` |
| 数据集与处理数据 | `opd_data/`、`dataset/`、`benchmarks/` |
| rollout | `opd_rollouts/`、`ocrd_anchor_train25_qwen3_1_7b_nonthinking_seed20260811_transformers_v2/` |
| 日志 | `opd_logs/` |
| 历史压缩包 | `archives_jsonl/`、`archive/`、`rollout_archive/` |
| 公共模型，只读复用 | `/storage/yukaichengLab/share/model/` |
| 公共数据集，只读复用 | `/storage/yukaichengLab/share/datasets/` |

`organized/` 提供分类入口，原始目录继续作为旧脚本的有效路径。不要把新的 H800 Kubernetes PVC 与此旧西湖 `/storage` 文件系统视为同一存储。
