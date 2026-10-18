# OpenLux GPT-4o-mini 数学网格重评分

已确认 prompt：system_prompt.txt。正式 Slurm job：105045。初始 job 105031 因 SSH 出网失败停止，未调用评分 API。

服务器输出：/storage/yukaichengLab/wangxin/test_eiloeo/opd_rescore/grid_math_gpt4o_mini_openlux_20260916

范围：MATH500 12,500 条、OlympiadBench 16,850 条、BBH 27,000 条，共 56,350 条。

每次请求仅含 candidate_answer 和 reference_answer；candidate_answer 复用源哈希匹配的旧 predicted 字段，旧记录未执行提取时重放旧提取器。允许旧提取器返回全文，不另行拼入 rollout。不向 judge 发送旧分数、配置名或模型身份。

- inputs.jsonl：冻结的逐条候选、标准答案、源哈希及实际用户消息。
- INPUTS_COMPLETE.json：输入数量和文件哈希校验。
- scores/<dataset>/<config>/<index>.json：is_correct 布尔判断、reason、实际模型名、调用状态、token 用量、输入及 prompt/config 标识。
- PROBE.json、PILOT.json：3 条 API 检查与 12 条真实记录试跑。
- progress.json：进度、并发上限和 API 错误计数。
- summary.json、summary.csv：每轮完成后汇总；未完成/API失败不计入已评分分母，并另列数量。
- COMPLETE.json：只有全部 56,350 条已处理且不存在 API 错误才写入。
- logs/job_105045.out：正式日志。

最大并发128，从16逐步增加；遇429减半并等待Retry-After。每条最多8次重试，失败记录另有两轮补跑。API失败保留 is_correct=null；空提取本地标 extraction_missing、is_correct=false。生成 trace_timeout 不自动判错，仍比较提取内容。

使用8CPU/24GiB，无GPU。内部出网服务只连接api.openlux.ai:443，随机token认证，TLS端到端校验证书。凭据文件权限0600，不写入代码、prompt、Slurm脚本或日志。作业退出会停止转发服务。

用量汇总为最终保存的成功评分响应，不包含API探针以及失败/未保存响应可能产生的费用。
