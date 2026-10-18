# Random position shift, greedy teacher and student

Submitted Slurm job **107976**, pinned to **gnho008**, on 2026-09-21 (replacement for preflight-only failure107970). Status RUNNING on gnho008; GPU preflight and all four seeded CUDA witnesses passed. Requests five H800 GRES, 32 CPUs, 256 GiB, UNLIMITED. Uses four healthy allocated GPUs as two split teacher/student pairs, 64 concurrent traces. The known failed GPU UUID GPU-c33c45c6-61d6-36bc-e5ac-6d80e3bc4968 (physical index 2) is excluded inside the allocation; selection fails closed if four healthy allocated devices cannot be identified. All selected cards must have <500 MiB used and >=74000 MiB free. Other jobs are untouched.

Same 4,571 frozen samples/eight datasets as shift5_k1_t2_20260918, one rollout each. Qwen3-1.7B student and Qwen3-14B teacher both generate with temperature=0, top_p=1, top_k=-1. Raw candidate scoring retains temperature=1. Earliest signed student-minus-teacher logprob gap strictly >2 triggers one teacher token at a reproducible random offset from {-5,-4,-3,-2,-1,1,2,3,4,5}. The shifted student token is banned. No intervention cap. Seed20260912 and shift_seed2026091802 unchanged. Left shifts clip after the previous teacher token or response start; right shifts extend until target, EOS or length limit. Actual shift may be zero at boundaries.

Non-thinking prompts, 64-token drafts, response40264/context40960 and 900-second cumulative trace budget retained. Independent fresh output root; no previous stochastic results reused. Generation only, matching the original submission scope.

Remote code: /storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/shift5_greedy_k1_t2_20260921
Remote output: /storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_ablation/shift5_greedy_k1_t2_20260921
Group: r1_shift5_greedy_k1_t2
Logs: /storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/shift5_greedy_k1_t2.107976.{log,err}

Validation: 12 tests passed locally/remotely, including shift boundaries, EOS, deterministic offsets, atomic rollback, fresh dispatcher and strict resume. Exact-byte equality checked for shift state machine, deadline, RPC, model binding and frozen prompt. AST verified greedy generation and unchanged raw scoring. Input SHA256 2c21e5857b8efbc41e1c5bb445d714e0dad41df39e0a7f28e93432aa8d9d64a1. Warm environment spec5ae61e04 unchanged. Runtime gates: allocated GPU selection, per-worker seeded CUDA witness, two-pair service checks for raw scoring, ban and seed-independent greedy generation, then two short shifted rollouts. Service and shifted rollout smoke checks gate formal dispatch.

GPU selection regression check passed: cgroup indices0..4 mapped by UUID to allocated physical0,1,2,4,6, then selected0,1,4,6.

Job107972 passed all CUDA witnesses but model inspection failed because this vLLM version parses GPU selectors as integers. Job-local uuid_compat/sitecustomize.py maps UUIDs to NVML indices for vLLM and its subprocesses; CUDA remains UUID-bound. Shared environment is unchanged. Replacement107976 passed model inspection and started model loading.

Launch verified:107976 RUNNING, submitted02:29:28/start02:29:40. All four models ready, both seed-independent greedy/scoring/ban service smokes passed, both actual shifted-rollout smokes passed, formal generation producing completed traces.
