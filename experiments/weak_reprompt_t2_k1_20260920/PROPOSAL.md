# 弱模型重提示单 token 分岔实验（待用户确认）

状态：用户已确认英文 v2 补全式 prompt，并授权单节点 8 卡提交。按最近全量口径运行 8 个数据集、4,571 题。

## 目的与基线

比较原阈值 k=1 强模型替换与弱模型在错误分支反馈条件下重新采样。新方法仍用强模型计算触发分数，不能称为完全无强模型方法。错误提示只是实验干预，不代表阈值已证明该位置错误。

基线实现参考 `experiments/shift5_k1_t2_20260918/run.py` 的候选打分逻辑及 `experiments/reverse_t2_k1_20260918/ablation_core.py` 的普通阈值状态机（只参考逻辑，不使用 reverse 实验的模型角色配置）。

## 精确定义

- Q：原始题目和原任务输出要求；保留原 system 约束。
- R：本轮开始前已接受的 response token IDs。
- D：弱模型本轮生成的完整 draft，最多 64 tokens，EOS 时可更短。
- j：D 中最早满足 log P_weak(D[j] | Q,R,D[:j]) − log P_strong(D[j] | Q,R,D[:j]) > 2 的位置。使用未经温度修改的实际 token logprob；等于 2 不触发。
- P = R + D[:j]：分岔点之前的完整已接受前缀，不包含待替换 token。
- F = D[j:]：旧分支，包括触发 token 及本轮剩余 draft。
- 保存 R、完整 D、j；未触发则接受整个 D。
- 触发时构建下述临时上下文，以弱模型最后一个输入位置的 logits 采样一个 token z。建议沿用基线 first-token ban，将 D[j] 的 logit 置为负无穷；temperature=1、top_p=1。
- 正式 response 更新为 P+[z]，丢弃 F。后续弱模型在原 Q + P+[z] 下继续普通生成；不保留临时纠错指令或错误分支。重新计算新轨迹的阈值位置，不复用旧位置。
- z 为 EOS 时正常结束。每次触发只替换一个 token；不限介入次数。

## 提议的临时 prompt（英文 v2，已确认）

根据用户修订：指令使用英文；在同一个 prompt 末尾直接写 New reasoning 和分岔点之前的前缀，从这里补全。原题和原推理保持原文，不翻译。

```text
Original problem:
{original problem and original output requirements}

The existing reasoning has a problem starting at the marked branching point. Keep the reasoning before that point and continue along a revised reasoning path.
The existing reasoning below includes the entire latest draft. The text after <BRANCH_POINT> is the old continuation to reconsider.

Existing reasoning:
{P}<BRANCH_POINT>{F}

Continue the unfinished New reasoning below exactly where it ends. Do not repeat its prefix or add a preamble. Follow the original problem's language and output requirements.

New reasoning:
{P}
```

实际输入严格结束在最后一份 P 的最后一个 token；代码块结束符不属于 prompt。在 New reasoning 的 P 后不加换行、分隔符、消息结束标记或新的 assistant 起始头。使用 token-ID completion 请求直接延续这个未完成的 prompt；不是先发送一个完整 user 消息再等待新的 assistant 回答。末尾 P 直接拼接原始 token IDs，以维持精确分岔边界；其余模板及引用文本先编码。一个 token 直接由输入末端 logits 采样，不先生成新的推理段。基线普通 rollout 的 non-thinking 格式保持不变；临时重提示使用上述显式 completion 格式，单独记录输入 token IDs。

## 数据与资源

默认沿用最近全量 R1 实验的 8 个数据集共 4,571 题：BBH 1,080；CRUXEval-I 800；CRUXEval-O 800；HumanEval+ 164；LiveCodeBench 175；MATH500 500；MBPP+ 378；OlympiadBench 674。每题一次 rollout，不是每题八次。历史 AIME24/25 不在这版 4,571 题清单中；若用户的“所有数据集”包含它们，则增加 60 题并核验原 prompt。

拟使用单节点 8 H800，4 组弱/强服务对；弱模型 Qwen3-1.7B 负责普通生成及重提示采样，强模型 Qwen3-14B 仅打分。沿用 R1 seed 20260912、原 dataset index、64-token draft、non-thinking、temperature/top_p=1/1、输出上限 40,264；冻结输入和模板哈希。重提示使用独立且可复现的 sample/round 派生 seed。

## 上下文与记录

临时 prompt 包含两份 P，可能提前超过现有 40,960 context 限制。建议首版保留该限制，完整上下文无法容纳时明确记为 reprompt_context_limit 并停止该题；不暗中截断旧推理，也不退回强模型替换。沿用 900 秒累计单题预算，单独报告超时及 context-limit 比例，避免将它们混为正常完成。该边界策略需随 prompt 一起确认。

记录每次原 draft、分岔位置、原 token、两模型 logprob/gap、新 prompt/token IDs（可内容寻址存储）、模板哈希、实际采样 token/概率、ban、seed、输入长度、EOS、重提示耗时及停止原因。额外统计重提示 prefill tokens 和总耗时，因为该方法能看到完整旧 draft，且计算预算与强模型 k=1 不相等。

确认后实施并验证：严格 >2、最早触发、完整 draft 引用、预填充 token 前缀保持、原 token ban、只替换一个 token、回到原上下文、EOS、上下文边界、失败原子性与断点续跑。通过 CPU 检查及作业内 GPU smoke 后运行全量。当前没有提交或预占 GPU。
