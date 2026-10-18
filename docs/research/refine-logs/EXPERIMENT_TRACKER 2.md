# 实验 Tracker：DeepMath Train-25% Top16 OPD Baseline

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| R001 | M0 | 参数化并静态核验 | Qwen3-1.7B <- Qwen3-14B，student Top-16 RKL，native non-thinking | train25 | py_compile、bash -n、manifest/config diff | MUST | COMPLETED | 2026-08-25；本地与远端均通过 |
| R002 | M0 | 提交前 fail-closed preflight | 数据 SHA、目标目录 quota probe、无重复 job | train25 | hash、write/read/remove、squeue/sacct | MUST | COMPLETED | SHA/manifest/env/16 MiB probe/target/duplicate 均通过 |
| R003 | M1 | 连续两步 exact-topology smoke | 单节点 exclusive 8×H800，4 student + 4 teacher | train25 前两步 | KL、grad、checksum、tokens、peak memory、ports、no think tags | MUST | COMPLETED | 93212 为节点启动故障，93467 在 step1 HF 导出遇 EFBIG；只将 HF shard cap 改为2GB并加 student-zombie fail-fast后，唯一 recovery smoke 93635 两步通过并 COMPLETED 0:0，checkpoint1/2及HF加载验证通过 |
| R004 | M2 | 正式一轮 OPD 训练 | global batch 64，222 steps，LR 1e-6 | 全部 14,208 padded rows | step KL/grad/tokens/finish reason | MUST | COMPLETED | 唯一恢复94033从ckpt133续跑并COMPLETED 0:0；最终step222 KL=0.12665620644274894、grad=0.5353502631187439、111788 tokens、零thinking标签 |
| R005 | M3 | 完整性与可恢复性验证 | CPU strict verifier | 五 checkpoint | exact set、pointer=222、HF load、optimizer/RNG present | MUST | COMPLETED | 93214自动释放并COMPLETED 0:0；checkpoint恰为44/89/133/178/222，pointer=222，HF 310 tensors真加载通过 |
| R006 | M4 | 五 checkpoint held-out 曲线 | native non-thinking fixed protocol | held-out 1000 | accuracy、Wilson、paired、difficulty、length | MUST | READY | 训练与严格验收已完成；后续单独确认/提交 |
