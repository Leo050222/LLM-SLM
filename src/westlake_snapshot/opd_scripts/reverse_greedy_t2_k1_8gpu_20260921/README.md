# Strong main / weak intervention, greedy, eight GPUs

Submitted Slurm job 107963 on 2026-09-21. One node, 8 H800, 64 CPUs, 512 GiB, UNLIMITED allocation; four split-GPU pairs, 128 concurrent traces. Excludes known unhealthy nodes 005/008/009/018. Reuses the unchanged warm verl-opd environment.

Qwen3-14B performs main reasoning; Qwen3-1.7B replaces one token at the earliest position where log P(token|14B) minus log P(token|1.7B) is strictly greater than 2. Original token banned, no intervention cap. Both models generate greedily (temperature 0, top_p 1, top_k -1). Raw scoring retains temperature 1. Legacy student/teacher names map to strong main/weak intervention.

Same 4,571 frozen prompts across eight datasets, one rollout each; SHA256 2c21e5857b8efbc41e1c5bb445d714e0dad41df39e0a7f28e93432aa8d9d64a1. Native non-thinking, chunk64, response40264/context40960, original 900-second cumulative per-trace deadline. Fresh outputs; no stochastic results reused. Generation only, matching original run scope.

Remote code: /storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/reverse_greedy_t2_k1_8gpu_20260921
Remote outputs: /storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_ablation/reverse_greedy_t2_k1_8gpu_20260921
Logs: /storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/reverse_greedy_t2_k1.107963.{log,err}

Validation: local/remote Python compile, shell syntax, exact-byte equality of state machine/deadline/RPC/model-binding with original experiment, generation/scoring sampling checks and preserved ban. Allocated-GPU free-memory preflight, seeded CUDA witness on every GPU, four-pair scoring/ban and seed-independent greedy service smokes gate dispatch. Actual runtime validation awaits allocation.

## Scheduling recovery
User requested trying 8h, 4h, then shorter walltimes. Job107963 started on018 at8h after exclusion removal; all8 CUDA witnesses failed (driver version12050 too old), zero generated results. Cancelled promptly and restored018 exclusion. Replacement107971 tried8h then4h pending, then2h successfully backfilled on029, start2026-09-21 02:32:40 CST/end04:32:40. All8 CUDA witnesses and4 greedy service smokes passed; actual TRACE_DONE outputs confirmed02:35. Current submission script2h. Per-trace900s unchanged.
