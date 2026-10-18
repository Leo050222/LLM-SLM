# Reverse intervention: strong main, weak intervention

Submitted Slurm job **106063** on 2026-09-18. One node, four H800 GPUs, 32 CPUs, 256 GiB, UNLIMITED time. Two resident model pairs, 64 concurrent traces, allocation retained until all generation completes. Reuses unchanged warm verl-opd environment; each GPU performs a seeded CUDA witness and each pair runs a service smoke before dispatch.

Main model: Qwen3-14B. Intervention model: Qwen3-1.7B. Threshold strictly > 2 on log P(main token | 14B) minus log P(main token | 1.7B). Earliest qualifying token is replaced by one weak-model token (k=1), with the original token banned; main model resumes. No intervention count cap and no external anchor judge. Internal legacy role names student/teacher mean main/intervention, respectively; manifests explicitly record this mapping and exact model paths.

One rollout each for 4,571 existing evaluation samples: cruxeval_o 800, cruxeval_i 800, humaneval_plus 164, mbpp_plus 378, livecodebench 175, olympiad_bench 674, bbh 1080, math500 500. Prompts are frozen from existing code and math/reasoning grids. Original indices retained for generation seeds. Native non-thinking, 64-token chunks, temperature/top-p 1/1, 40,264 output token budget, context 40,960, existing 900-second per-trace timeout preserved and explicitly marked. This allocation generates rollouts; external answer scoring is not included.

Frozen prompt SHA256: 2c21e5857b8efbc41e1c5bb445d714e0dad41df39e0a7f28e93432aa8d9d64a1.

Remote code: /storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/reverse_t2_k1_20260918
Remote output: /storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_ablation/reverse_t2_k1_20260918
Slurm logs: /storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/reverse_t2_k1.106063.{log,err}

Local validation passed: Python compilation, shell syntax, deterministic state-machine check for strict threshold, role handoff, k=1, first-token ban, and EOS. Actual GPU runtime witness awaits allocation.

## Recovery 2026-09-18

Original job 106063 failed after 90 seconds with zero completed samples. Both model-pair service smokes passed. Dispatcher incorrectly required an unrelated recovery_4gpu_20260917/migration.json. Replaced that migration helper with fresh manifest creation and strict identical-manifest resume validation. Tests passed for fresh creation, identical resume, and rejection of mismatched configuration. Confirmed no existing manifest/results or active same-name job before resubmission. New job: **106157**, original experimental parameters unchanged. New Slurm logs: reverse_t2_k1.106157.{log,err}.

## Resume job 106181

Job 106157 failed after 17m47s with 1,630 saved results (5 trace timeouts). Weak-model top16 diagnostic extraction found only 12 valid candidates and killed its worker. No-judge threshold rollout never consumes candidate alternatives; removed their extraction for both roles while preserving exact actual-token logprobs, sampling requests and trigger rules. Tested candidate construction with empty diagnostic arrays and strict >2 boundary. Archived original manifest as manifest.before_topk_fix_106157.json; verified only worker.py and smoke.py hashes changed, updated corresponding code identities, preserved scientific settings and all results/checkpoints/cumulative timings. Submitted continuation 106181.

## Completion

Verified complete at 2026-09-18T15:14:53.888354+08:00. Job 106181 COMPLETED 0:0, elapsed 03:51:41. All eight datasets covered, 4,571/4,571 saved outputs: 4,327 normal stop and 244 trace_timeout (900-second censoring). ALL_GENERATION_COMPLETE.json and group VERIFIED.json present; scoring has not been performed by this generation job.
