# Weak reprompt t2 k1

Authorized English raw-completion prompt ends in `New reasoning:\n` followed by the exact token IDs of the accepted prefix. Full latest draft is quoted in Existing reasoning. At the earliest signed weak-minus-strong logprob gap >2, Qwen3-1.7B samples one token from this new context, with the original entry token banned, then resumes under the original task context. Qwen3-14B only scores. No shifted position or anchor judge.

Job 107636: one node gnho031, eight H800, 64 CPU, 512 GiB, four model pairs, 128 concurrent traces. Eight datasets, 4571 rows, one rollout each, same frozen inputs and R1 seed 20260912. Draft 64, temperature/top_p 1/1, max output 40264, context 40960, cumulative per-trace timeout 900 seconds. Reprompt context overflow stops explicitly without truncation. AIME is outside this frozen dataset set. Generation only; no external scoring job submitted.

Seven CPU tests passed locally and remotely. Allocation preflight passed on all eight GPUs with 0 MiB initially used. Each worker runs a seeded CUDA witness; all four service smokes and forced-trigger reprompt smokes must pass before full dispatch. Check runtime status in LAUNCH.json and the job logs.

Remote code: `/storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/weak_reprompt_t2_k1_20260920`.
Remote output: `/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_ablation/weak_reprompt_t2_k1_20260920`.
