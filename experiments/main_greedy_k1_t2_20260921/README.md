# Main method, k=1, pi=2, both greedy

Slurm 108123 started 2026-09-21 12:10:47 CST on gnho017: 8 H800, 4 split model pairs, 128 concurrent traces, 64 CPU, 512 GiB, unlimited allocation.

Full frozen main evaluation: CRUXEval-O 800, CRUXEval-I 800, HumanEval+ 164, MBPP+ 378, LiveCodeBench 175, OlympiadBench 674, BBH 1080, MATH500 500. Total 4,571 prompts, one rollout each. Input SHA256 2c21e5857b8efbc41e1c5bb445d714e0dad41df39e0a7f28e93432aa8d9d64a1 matches existing ablations.

Qwen3-1.7B student and Qwen3-14B teacher both use temperature=0, top_p=1, top_k=-1. Raw scoring retains temperature=1. At the earliest signed student-minus-teacher logprob gap strictly >2, replace one token with teacher output, banning the original student token. No anchor judge, no intervention cap, no position shift. Native non-thinking prompts, chunk64, response40264/context40960, seed20260912 inherited from main R1; cumulative per-trace deadline900s, timeout partial outputs preserved and counted wrong. No stochastic traces reused.

Remote code: /storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/main_greedy_k1_t2_20260921
Remote output: /storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_ablation/main_greedy_k1_t2_20260921
Group: r1_main_greedy_k1_t2
Logs: /storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/main_greedy_k1_t2.108123.{log,err}

Deployment reuses verified reverse-greedy infrastructure with normal student/teacher model binding. Core rollout, worker, deadlines, RPC and strict-resume migration are byte-identical; only input/output identities and model roles change. Local/remote Python and shell syntax pass. Warm environment spec5ae61e04 unchanged. All eight allocated GPUs passed free-memory preflight. Seeded CUDA witnesses and four pair service checks gate formal dispatch. Answer grading is separate and has not been submitted.

Startup verified: all eight seeded CUDA witnesses and four greedy/raw-score/token-ban pair smokes passed. Formal generation snapshot677/4571 completed,1589 checked interventions satisfy strict gap>2,k1,and original-token ban;0 TRACE_ERROR. Runtime manifest confirms correct1.7B student/14B teacher binding.
