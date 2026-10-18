# 内部并发可行性分析

结论：可行，优先考虑每GPU单一AsyncLLM引擎、逐请求完成返回和立即补入，先保持max_num_seqs=8，再独立测试16。当前已存在外层并发，单纯增加线程不足以解决引擎入口处的整批等待。只读分析，未改运行代码、重启任务或申请额外GPU；没有GPU A/B实测加速倍数。

## 当前结构与证据

- 每节点4对Student/Teacher，128轨迹线程，每对32轨迹。Backend.candidates已并行提交双方评分。
- 每GPU唯一worker，batch_size=8且max_num_seqs=8。worker同步调用engine.generate，等其返回全部输出后逐个整理/写入结果，再claim下一批。
- 实际环境Python3.11.15，vLLM路径miniconda3/envs/verl-opd/lib/python3.11/site-packages/vllm。offline_utils.py:573的_run_engine循环至has_unfinished_requests为False，才整体返回；证实短请求也被应用层整批等待挡住。
- 实际async_llm.py:232/280/524/709含from_engine_args/add_request/generate/abort；generate有每请求结果队列与独立输出处理任务，取消协程时会abort对应内部请求。output_processor.py包含FINAL_ONLY和prompt_logprobs累积/返回路径。
- 最新queues.json：代码教师各GPU有25/31/31/28个请求文件；某卡前缀69–15811token混合。数学教师有23/25/16/5个请求。请求文件可能包含正在执行的当前批次，因此这些数量不是精确未执行队列深度，但大于8的部分证明存在待补入工作。

## 建议方案

A：异步入口，保持相同并发上限8。

每GPU一个AsyncLLM，一条提交循环加每请求异步结果任务。请求完成后立即发布对应结果，空位立即补入。文件扫描、锁、读写移出事件循环，使用有界IO执行器；不要在事件循环里做大段同步JSON处理。不得并发调用同一同步LLM.generate来模拟此设计。

B：在A验证后，测试max_num_seqs=16及明确的token预算，再按内存/吞吐决定是否需要32。外层128轨迹先保持不变，避免混杂。不能认为8→16必定翻倍；prompt logprobs和长上下文可能成为计算、CPU输出或显存瓶颈。

C：视测量结果再做节点内socket/IPC替代共享盘热路径，持久输出和检查点保留共享存储。更深的Teacher服务共享池/资源配比调整暂不与A混合。

## 必须处理的兼容点

1. 原claim_batch把请求文件留在requests目录，异步扫描必须维护waiting/inflight/done状态并排除inflight，避免重复提交。同请求多个等待者共享一个引擎request_id。
2. 最后一个有效租约过期才abort。一个等待者超时不能取消别人；完成/取消竞态、唯一请求ID及结果错配需要专门测试。
3. 保留FIFO公平性、有界inflight数量和失败传播，不能无限向engine灌请求。长短请求交错时要监测score/generate两类的p95等待。
4. 沿用exact token IDs、SamplingParams、seed、prompt_logprobs=16、Student实际token logprob和Teacher top16、首token禁止、EOS、累计900秒。异步返回按request_id绑定，禁止按完成顺序zip回原输入。
5. 使用FINAL_ONLY要真实验证prompt_logprobs完整位置对齐、Teacher top16与实际token覆盖；现有源码具备返回路径不等于已通过本任务数值验证。
6. 不顺带打开kv-sharing-fast-prefill：v0.23源码明确拒绝它与prompt_logprobs结合。普通前缀缓存是另一项需单独验证的优化。
7. 同seed在不同批次布局下未必逐token完全一致。比较相同前缀的评分数值与阈值决策，并保留现有批次布局自身的重复性对照。900秒含排队，最终必须记录新runtime版本及超时分布变化。

## 哪些不能直接并发

同一轨迹的下一轮依赖本轮阈值判断和Teacher替换后的前缀，不能提前正式执行；Teacher桥接首token与后续k-1token也有前缀依赖。提前推测执行属于额外算法工程，会浪费被丢弃路径的计算。本方案主要提高不同轨迹之间的流水重叠，不改变单轨迹因果顺序。也不在每张卡加载第二份14B模型。

## 最小验证设计

依次比较：现有FIFO同步8 → 异步8 → 异步16。先对固定混合请求集回放，覆盖score/1token桥接/64token学生生成/长短上下文；各方案同一请求集、token IDs、seed，独立结果缓存，相同预热。请求回放只测服务能力；另以相同分层题目集跑完整闭环rollout验证端到端收益。包含k1/k8/k32、各数据集和长轨迹。

主要指标：有效完成轨迹/小时、有效输出token/s、每类请求p50/p95等待、timeout率、OOM/错误、GPU利用率和host输出处理时间。至少重复3次、随机化A/B顺序，不能仅凭GPU util宣布提升。保持总资源不变，可在分段边界使用一对现有GPU做小规模验证，通过后再迁移正式任务。此次未实施该GPU验证，也不据此调整当前ETA。

## 官方依据

- https://docs.vllm.ai/en/v0.23.0/api/vllm/v1/engine/async_llm/
- https://docs.vllm.ai/en/stable/api/vllm/config/scheduler/

本地可核查位置：experiments/grid_fifo_20260916/worker.py、dispatch.py、queue_rpc.py；experiments/nojudge_grid_20260912/run.py的Backend.candidates。
