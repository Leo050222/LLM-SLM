# 第 6 个 token logits 重放结果

作业 106101 已完成（COMPLETED 0:0，16 秒），gnho020，1 张 H800。
GPU UUID: GPU-f92d94b2-4d4d-5832-3276-2667bcd9648c。作业结束后释放 GPU。
模型 Qwen3-1.7B，BF16，Transformers 5.6.0 / PyTorch 2.11.0+cu130，SDPA，batch=1，非思考模式。

随机种子 20260918，抽到 CRUXEval-I sample_315。原 prompt 77 tokens，生成 5 tokens 后前缀 82 tokens。
输出 token IDs: [58, 29421, 60, 8645, 1131]，文本为 `[INPUT] challenge='`。
原始增量生成在喂入第 5 个输出 token 后提取 logits，因此预测的是第 6 个 token。
第二次请求从保存的完整 82 个 IDs 原样读取，全量 prefill，不继承第一次的 KV cache。

| 指标 | 结果 |
|---|---:|
| 全词表逐元素/字节完全一致 | 否 / 否 |
| 不同位置 / 词表大小 | 111225 / 151936 |
| 最大绝对差 | 0.25 |
| 平均绝对差 | 0.0360590816 |
| RMS 差 | 0.0487674922 |
| argmax 一致 | 是，47772（czy） |
| softmax 概率最大绝对差 | 0.00000500679 |
| 两次 fresh prefill 字节完全一致 | 是 |
| 两次相同增量路径字节完全一致 | 是 |

结论：在本样例和配置下，相同 token IDs 不保证增量 decode 与全量 prefill 的 logits 完全一致。
两种路径各自重复都精确一致，支持差异与计算路径/数值舍入有关的解释；本实验没有定位到具体算子。
结果只适用于本次 Transformers SDPA BF16 单样例，未测试 vLLM 或其他精度。

已保存四份完整原始 logits 张量、全部 prefix IDs、样例、版本和指标。
本地独立读取 PT ZIP 中 BF16 存储，复核不同元素数、最大绝对差及两组对照的字节相等，均通过。
