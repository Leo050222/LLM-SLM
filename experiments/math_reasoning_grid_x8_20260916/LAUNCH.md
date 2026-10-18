# Math/reasoning grid: retain one rollout and add seven

Authorized 2026-09-16: add one node with eight H800 GPUs for the remaining datasets while code-grid job104980 continues independently. Combined resource ceiling for these two experiments: two nodes, sixteen H800 GPUs.

Scope: MATH500 500, OlympiadBench 674, BBH 1080; total2254 prompts. Thresholds2/2.5/3/3.5/4 crossed with k1/4/8/16/32. Historical replicate0 uses seed20260911; new replicates1–7 use seeds20260912–20260918. 175 new groups,394450 new outputs; combined target450800 including56350 historical outputs. No baseline repeats or AIME expansion.

Remote old results: `/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_hard/nojudge_grid_20260912`.

Remote new results: `/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_hard/math_reasoning_grid_x8_20260916`.

Remote scripts: `/storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/math_reasoning_grid_x8_20260916`.

Old output identities, prompt tokens, problem IDs, complete status, no-judge state and per-group coverage are checked before submission; ordered content hashes are retained in replicate0_audit.json. Old outputs are read-only and not copied into new replicate directories.

Qwen3-1.7B student/Qwen3-14B teacher; native non-thinking, temperature/top-p1/1, student chunks64, response/context limits40264/40960, strict signed student-minus-teacher logprob gap, unlimited interventions, first-token ban, no judge. Historical random_rates metadata is preserved in settings, although unused in nojudge_grid mode. Uses the current validated worker, which omits unused Student top16 serialization; generated tokens and Teacher top16 logic are unchanged from the historical algorithm.

Stopping-policy caveat: historical replicate0 is mixed: early groups had no900s deadline; late groups adopted cumulative900s, with some already-completed legacy rows. New seven replicates uniformly use900s cumulative active trace wall time including RPC waits, preserve partial outputs and flag timeout incorrect. Retained replicate0 is not identically censored; analyses must disclose this difference. Four pairs versus historical mixed pair counts can also affect timeout rates.

Runtime: one exclusive node, typed gpu:H800:8, requested64CPU/512GiB, four model pairs/128 concurrent traces. Six-hour Slurm segments stop admitting traces after five hours; each successful unfinished segment submits one afterok successor. Failures stop the chain for diagnosis; output/checkpoint/timing files remain resumable. Check job_ids.txt and submission.json before any recovery submission. ALL_GENERATION_COMPLETE.json requires all175 groups verified. No CPU scoring job is part of this rollout launch.

Checks: Python/shell syntax passed, distinct first-round seeds across all8 replicates for every2254 question, runtime worker files identical to validated code-grid launch. Warm environment spec hash remains5ae61e04e36b0afd56a5b580951741b3f4b16813c0a1acb521dfa0691c3e8107; no environment rebuild.

Submitted job104982, RUNNING on gnho032 at2026-09-16 16:10:21 Asia/Shanghai. One node, eight typed H800s; all eight devices showed0MiB before worker launch. Code job104980 remains RUNNING on gnho031. All56350 old rows passed identity/coverage checks.

Launch validation passed: all eight CUDA_WITNESS_OK markers, all eight model identities ready; dispatcher produced COMPLETE outputs with finish_reason=stop, beyond model-loading status.

## FIFO recovery 2026-09-16
Current job: **104997**. Original job stopped under user-authorized efficiency repair. New common runtime at opd_scripts/grid_fifo_20260916, with per-experiment code.slurm/math.slurm successors. Consult results/recovery_fifo_v2/submission.json and job_ids.txt; initial submission.json is historical. FIFO homogeneous batches, independent caller leases, orphan cleanup. Old complete outputs and900s cumulative timers preserved. Four live service smoke pairs passed before formal generation. Migration and tests passed; detailed report at reports/grid_fifo_fix_20260916/REPORT.md.

## Async trial rejected; current FIFO job105032
2026-09-16: user authorized implementing/testing async8. Initial105028 benchmark input-ID lookup failed and automatically resumed FIFO; corrected105032 completed3 trials but async RPC throughput was0.60–0.66x baseline and numerical gate failed. No async production rollout was launched. Current105032 ongnho032 runs original grid_fifo_20260916/node.sh after fallback; future successor uses grid_fifo_20260916/math.slurm. All4 service smoke pairs passed and TRACE_DONE confirms continued production. Code job104996 was never interrupted. See reports/grid_async_trial_20260916/REPORT.md.

## Four-GPU resume 2026-09-17
User requested releasing four GPUs for other users. The prior request was interrupted before execution; at00:46 job105164 still used8H800. Cancelled105164 and submitted105199 using `opd_scripts/grid_math_4gpu_20260917/math.slurm`: non-exclusive4H800,32CPU,256GiB, two model pairs/64 concurrent traces. Successor submissions use the same4GPU script. Preserved35348 completed outputs; exact SHA256 values and original manifests recorded in remote `recovery_4gpu_20260917`. Traces and cumulative900s timing files remain untouched. Migration permits execution-only changes, rejects seed/scientific changes, and passed replay/idempotence/seed-rejection checks for175 manifests. Worker and inference logic match the original FIFO version. Smoke now derives the pair count from GRID_REPLICAS. Code job105134 remains on8H800.

## User correction: restored eight GPUs, 2026-09-17
User explicitly corrected the4GPU request during launch verification. Job105199 successfully produced formal outputs before cancellation; it is now stopped. Resume job105201 uses `opd_scripts/grid_math_8gpu_20260917/math.slurm`,8H800/64CPU/512GiB, four model pairs/128 concurrent traces, with8GPU successors. The4GPU phase added102 outputs;35450 completed outputs are preserved and hashed under `recovery_8gpu_20260917`. All175 source manifests passed resource-only migration, idempotence and scientific seed-change rejection checks. No trace/timing/output edits were made. Original FIFO worker, dispatcher, budget and queue implementations are byte-identical. Previous4GPU handoff is superseded by this correction.
