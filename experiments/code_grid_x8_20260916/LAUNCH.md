# Code grid: eight rollouts per question and configuration

User request: one node, eight H800 GPUs; retain the previous rollout and add seven.

- Initial Slurm job: **104980**, RUNNING on **gnho031** (2026-09-16 16:01:41 Asia/Shanghai).
- Grid: thresholds 2, 2.5, 3, 3.5, 4 × k 1, 4, 8, 16, 32.
- Dataset: original 2,317 prompts, unchanged token IDs; CRUXEval-I/O, HumanEval+, MBPP+, LiveCodeBench.
- Retained replicate 0: seed 20260911, 57,925 outputs. All outputs were read and verified against the prompt/configuration identity, with ordered output hashes saved in replicate0_audit.json. Historical files remain in their original directory and were not changed.
- New replicates 1–7: seeds 20260912–20260918, 175 groups, 405,475 new outputs. Combined target: 463,400 outputs.
- Only the 25 intervention configurations are repeated; student-only and teacher-only baselines are outside this request.
- Four student/teacher pairs, 128 concurrent traces (32 per pair), single exclusive node with typed gpu:H800:8; 512 GiB requested RAM.
- Original generation worker and algorithm reused. Dispatcher changes only add per-replicate seed/metadata and stop admitting traces after five hours to allow clean segment shutdown.
- Native non-thinking; temperature/top-p 1/1; student drafts 64 tokens; output/context limits 40264/40960; signed student-minus-teacher gap strictly above threshold; unlimited takeovers; first replacement token ban; no judge.
- Original 900-second cumulative active trace deadline, including RPC waits, is retained. Timeouts keep committed partial responses and are flagged incorrect. Parallel execution increases from two pairs in the historical run to four pairs, so wall-time-based timeout rates can differ.
- Each successful incomplete segment submits exactly one afterok successor with the same one-node/eight-GPU request. Failures stop the chain for diagnosis; completed traces and cumulative timing files persist. No extra GPU jobs or CPU scoring jobs were submitted.

Remote results: `/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_code/code_grid_x8_20260916`.

Remote scripts: `/storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/code_grid_x8_20260916`.

Historical replicate: `/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_code/code_reasoning_grid_20260913`.

Submission is guarded by submission.lock/submission.json. Check job_ids.txt for subsequent segments; do not resubmit the initial job blindly. ALL_GENERATION_COMPLETE.json is written only after all 175 groups pass the complete-output verifier.

Validation: shell/Python syntax checks passed; all 2,317 question IDs have distinct initial round seeds across eight replicates; environment specification canonical SHA remains 5ae61e04e36b0afd56a5b580951741b3f4b16813c0a1acb521dfa0691c3e8107 (warm reuse, no package changes). All eight CUDA_WITNESS_OK markers passed on NVIDIA H800. Launch-time GPU memory was 0 MiB on all eight devices. All eight model identities are ready; dispatcher entered formal generation, and TRACE_DONE entries show actual completed outputs with finish_reason=stop. Job104980 remained RUNNING on gnho031 at elapsed2:17.

## FIFO recovery 2026-09-16
Current job: **104996**. Original job stopped under user-authorized efficiency repair. New common runtime at opd_scripts/grid_fifo_20260916, with per-experiment code.slurm/math.slurm successors. Consult results/recovery_fifo_v2/submission.json and job_ids.txt; initial submission.json is historical. FIFO homogeneous batches, independent caller leases, orphan cleanup. Old complete outputs and900s cumulative timers preserved. Four live service smoke pairs passed before formal generation. Migration and tests passed; detailed report at reports/grid_fifo_fix_20260916/REPORT.md.
