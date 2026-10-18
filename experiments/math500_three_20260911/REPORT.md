# MATH500三组non-thinking单次rollout结果

三组均500题，每题一次。使用同一MATH500数据、原始Qwen3-1.7B student和Qwen3-14B teacher、同一non-thinking提示、temperature=1/top_p=1；最大生成40264 token。

|实验|做对题数|正确率|相比student|
|---|---:|---:|---:|
|Student直接生成|364/500|72.8%|—|
|在线teacher每次8 token，不限介入次数|407/500|81.4%|+8.6个百分点|
|Teacher开头最多256 token后student续写|376/500|75.2%|+2.4个百分点|

在线比固定前缀高6.2个百分点。在线相对student新增做对62题，同时失去19题，净增43题；相对固定前缀新增做对52题，同时失去21题，净增31题。这是单轮结果，不宣称多seed稳定性。

在线介入共1125次（平均2.25次/题，单题最多47次），teacher实际写入8997 token（平均17.994/题），judge调用5406次。固定前缀teacher写入125980 token（平均251.96/题），47条teacher自然结束后直接保留完成答案。teacher写入量不代表总推理成本：在线另有teacher候选评分和judge调用。

生成102194于gnho011正常完成（18分32秒）；随后评分102192正常完成（10秒）。三组生成全部结束后才启动评分。评分采用本地EvalScope MATH500 extract_answer + numeric acc中的math_equal；1500条全部评分ok，无missing、timeout或error。

核验：1500个题目ID、answer字段、共享提示token、COMPLETE状态、评分source SHA256；teacher前缀连续性；在线每次bridge不超过8 token；全部评分晚于全量生成完成标记。完整输出与评分已同步到results/并逐条重新验证哈希。

注意：此前MATH500 90.4%为thinking模式，不混入本表。
