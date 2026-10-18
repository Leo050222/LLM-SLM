# 实验计划：DeepMath Train-25% Student-Top16 Reverse-KL OPD Baseline

**问题**：在每个难度抽取 25% 的 DeepMath difficulty>=6 训练集上，训练一个与现有 native non-thinking 基线协议完全匹配的 Qwen3-1.7B OPD baseline，并保存五个可恢复 checkpoint。  
**方法主张**：使用当前 student on-policy rollout 状态上的 student Top-16 stop-gradient 支持集，最小化重归一化 `KL(student || teacher)`，应当优于未微调的 Qwen3-1.7B，同时保持方法为最小 token-local OPD baseline。  
**日期**：2026-08-25  
**状态**：仅完成提交前设计；未提交任何 Slurm 任务。

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|---|---|---|---|
| C1：token-local student-Top16 Reverse-KL OPD 能提升 Qwen3-1.7B | 它是后续 recovery-span 方法必须超过的强 baseline | 固定 held-out 1000 上，至少一个预注册 checkpoint 明显高于原始 1.7B non-thinking `283/1000=28.3%`；协议、数据和 scorer 完全一致 | B1, B2, B3 |
| C2：收益不是 checkpoint cherry-pick 或 generation-mode 改变造成 | 五个等进度 checkpoint 才能看出真实训练曲线 | 五个 checkpoint 全部保存并按同一测试协议评测；报告完整曲线而非只报最佳点 | B2, B3 |
| Anti-claim：收益来自 thinking、额外 reward、teacher forcing 或数据变化 | 这些混杂会破坏 baseline 的解释力 | 训练与评测均 native non-thinking；无 reward/SFT/CE；数据 SHA、顺序 SHA、采样配置和 objective 写入 manifest | B1, B2 |

## Paper Storyline

- 主文必须证明：最小 student-Top16 token-local OPD baseline 在相同 non-thinking 协议下的学习曲线与 held-out 效果。
- 附录可支持：逐难度曲线、训练 KL/梯度/长度统计、五 checkpoint 的 paired 比较。
- 本轮刻意不做：bridge、anchor、replay、reward mixing、full-vocabulary KL、Top-K ablation、多 seed；这些不是本 baseline 的组成部分。

## 实验 Blocks

### B1：数据与协议冻结

- Claim tested：排除数据、thinking mode 与 sampling 的混杂。
- Dataset：
  - 训练：`deepmath_ge6_opd_train25pct_perdifficulty_seed20260823/train.parquet`
  - 14,208 行 = 14,158 unique + 50 明示 padding；恰好 `222 × 64`
  - SHA-256：`fdcb58732d22f5ab27d111f07c8746836d983c5d15d1d800970154ea1f3cfc91`
  - 测试：固定不相交 held-out 1000，SHA-256 `6ef8e66d181b7522bad2904ad1fcf20d8fc152af0c9d7eb05e07a9f2f0415ae0`
- 模型：student `/storage/yukaichengLab/share/model/Qwen3-1.7B`；frozen teacher `/storage/yukaichengLab/share/model/Qwen3-14B`；tokenizer 固定为原始 Qwen3-1.7B。
- 生成协议：`enable_thinking=False`，temperature `1.0`，top-p `1.0`，generation top-k `-1`，seed `20260811`，`max_model_len=40960`，`max_response_tokens=40264`。
- 成功标准：manifest 同时记录 data/model/tokenizer/config/order SHA；生成 response 中无 `<think>`/`</think>`。
- Priority：MUST-RUN。

### B2：Student-Top16 Reverse-KL 训练

- Claim tested：建立原始 token-local OPD baseline。
- On-policy 单步流程：
  1. 当前 student 对该步 64 题自由 rollout；
  2. 在每个采样 prefix 上取当前 student 未修改分布的 Top-16 token IDs；support 选择 stop-gradient；
  3. student 与 frozen teacher 在同一 16-token support 上计算 logits，并分别重归一化；
  4. 最小化 token-weighted `KL(student || teacher)`：`sum_i p_S(i)(log p_S(i)-log p_T(i))`；
  5. 更新 student 后导出新 HF snapshot，下一步重新 rollout。
- 明确排除：task reward、policy-gradient surrogate、SFT/CE、teacher-forced trajectory、loss mixing、bridge/replay。
- 优化配置：一轮训练，global batch `64`，总步数 `222`，AdamW，LR `1e-6`，weight decay `0.01`，betas `(0.9,0.999)`，global grad clip `1.0`，BF16 autocast，student train state FP32。
- Checkpoints：预注册在 `44, 89, 133, 178, 222`，对应约 20/40/60/80/100% 数据进度；必须恰好五个。
- 每个 checkpoint：HF safetensors、完整 model state、optimizer state、torch/CUDA RNG、metrics、manifest；`latest_checkpointed_iteration.txt` 原子更新。
- 断点恢复：只允许从最新完整预注册 checkpoint 恢复；任何更新后的 metrics 或 partial step 必须归档到 `attempts/` 后重算，不得从半步或仅 HF 权重续跑。
- 持久化策略：保留 compact per-step metrics、五 checkpoint 和失败 attempts；逐 token teacher logits、Top-16 tensors 与 step traces 在该步完成后删除，避免额外 quota 压力。
- 成功标准：222 步完成，无 NaN/OOM/NCCL/端口/存储错误；每步 KL 有限且非负、梯度有限、参数 checksum 非零变化；五 checkpoint 完整且最终 HF 可加载。
- Priority：MUST-RUN。

### B3：五 checkpoint 的固定 held-out 评测

- Claim tested：C1 与 C2。
- 评测协议：与现有原始 1.7B/14B native non-thinking held-out 结果完全一致；同一 1000 题、同一 tokenizer、sampling、长度、seed 和 rule scorer。
- 比较对象：原始 1.7B `28.3%`；14B teacher `43.6%` 仅作 capacity reference；五个训练 checkpoint 全部报告。
- 主指标：accuracy、Wilson 95% CI、相对 1.7B 的 pp 变化、paired outcomes；次指标：逐难度 accuracy、response length、finish reason。
- 成功标准：五点曲线完整；不因只选择最佳 checkpoint 而省略其它点。
- 本轮提交边界：训练完成后另行提交评测链，不隐含在首次训练提交中。
- Priority：MUST-RUN，但在训练完成后执行。

## 用卡与调度方案

### 推荐：单节点独占 8×H800

- Slurm：`--nodes=1 --gres=gpu:H800:8 --exclusive`，明确排除已知不兼容 `gnho009,gnho018`；CUDA kernel preflight 通过后才允许写正式输出。
- GPU 0–3：四个 persistent Qwen3-1.7B student data-parallel ranks；每 rank 每步处理 16 题。
- GPU 4–7：四个 frozen Qwen3-14B teacher workers；每 rank 对应同一批 16 条 student trace。
- rollout 时 student daemon 与本 rank vLLM 共置于 GPU 0–3；沿用已验证的低 vLLM memory utilization、独立 Triton/Inductor cache、每 rank/step 探测空闲端口。
- 优点：只有整台机器完全空闲时才启动，不在多台机器上穿插单卡；无跨节点 NCCL；比旧 2 节点 16 卡拓扑更节制。
- 代价：相对已验证的 8-student/8-teacher 两节点拓扑，每 rank 工作量翻倍；预计实际训练约 6–10 小时，申请 24 小时 walltime 留出长尾与 checkpoint I/O 余量。队列等待不计入该估计。
- 预计 formal 资源：约 48–80 H800 GPU-hours；两步 exact-topology smoke 预计 10–25 分钟。

### 不推荐默认使用：两节点 16×H800

- 它会更快，但资源翻倍并重新引入跨节点 NCCL；除非单节点 8 卡 smoke 显示吞吐或显存不可接受，否则不使用。
- 若未来确需 16 卡，也必须申请两台完整 8 卡节点并 `--exclusive`，绝不使用跨多机的零散单卡数组。

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|---|---|---|---|---|---|
| M0 | 实现与静态核验 | 参数化 `thinking` 与单节点 4+4 launcher；py_compile/bash -n；只读数据/hash/无重复任务检查 | 配置 manifest 与冻结协议逐项一致 | CPU-only | hard-coded thinking 或旧 15% 路径残留 |
| M1 | 两步生产拓扑 smoke | 单节点独占 8×H800，exact 4+4 topology，连续两步 | 两步均 finite KL/grad、checksum 改变、端口轮换、无 think tags、checkpoint 1/2 可恢复 | 约 1.5–3.5 GPU-hours | 共置显存、端口或长 trace 问题 |
| M2 | 正式训练 | `afterok:M1` 的唯一 222-step job | 只有 smoke 完整成功才释放；不得重复提交 | 48–80 GPU-hours | quota、长尾、节点中断 |
| M3 | 严格验证 | afterok CPU verifier | 五 checkpoint 恰好完整、pointer=222、HF load/resume state 有效 | CPU-only | checkpoint I/O 不完整 |
| M4 | 五 checkpoint 评测 | 后续独立单节点评测链 | 五个 checkpoint 均 1000 条完整结果 | 后续估算 | 评测资源与训练混用 |

## Compute and Data Budget

- GPU：推荐单节点 8×H800；不申请散落卡。
- 正式训练预计：6–10 小时，48–80 H800 GPU-hours；walltime 24 小时。
- 持久存储：每个完整 checkpoint 约 24 GiB，五个约 120 GiB；run metrics/attempts 预留约 5–10 GiB，总计提交前至少确认约 130 GiB quota 余量。
- 当前全局 `/storage` 余量为 47 TiB、inode 1%，但此前出现过用户/项目 quota 型 `ENOSPC`，因此确认后提交前仍必须在目标目录做 create/write/read/remove probe，并拒绝在 probe 失败时提交。
- 最大瓶颈：逐步重新加载 vLLM/teacher 与四 rank 每 rank 16 条 trace 的长尾，而非模型显存。

## Risks and Mitigations

- **脚本当前 hard-code `thinking=True`**：确认后把 thinking mode 参数化，manifest、chat template 与 verifier 同时 fail-closed 检查 `False`。
- **单节点 4+4 尚未做完整训练验证**：先跑连续两步 exact-topology smoke；formal 只能 afterok 释放。
- **端口复用/EADDRINUSE**：每 step/rank 使用确定性 seed port，首次 bind 探测后沿 rank 互斥余数类寻找空闲端口，并记录 selected port。
- **共置显存风险**：沿用 student daemon + vLLM 已验证的 0.30 utilization、chunk=256、BF16；smoke 必须记录每 rank peak memory。
- **存储 quota**：提交前目标目录写探针；只保留 compact metrics 与五个完整 checkpoint，逐 token临时文件按步清理。
- **失败恢复跳数据**：冻结 `train_order.json` SHA；只从最新完整 checkpoint 恢复，后续 partial artifacts 先归档再重算。
- **资源礼仪**：只申请单节点完整 8 卡并 exclusive；不创建 8/16 个散落单卡数组。

## Final Checklist

- [x] 固定同一 25% 训练集及 SHA
- [x] 固定 native non-thinking 作为推荐协议
- [x] 固定 student-Top16 stop-gradient Reverse-KL 目标
- [x] 固定 222 steps、LR、batch 和五 checkpoint 位置
- [x] 固定单节点独占 8×H800 的用卡策略
- [ ] 用户确认配置，尤其确认 `enable_thinking=False`
- [ ] 实现参数化与单节点 launcher
- [ ] 两步 exact-topology smoke 通过
- [ ] 唯一 formal job 提交
- [ ] 五 checkpoint 严格验证
- [ ] 五 checkpoint held-out 评测

