# CRUXEval-I fresh-prefix cap-1 replication

Authorized scope: 100 new CRUXEval-I problems with an observed first threshold > 2 anchor, excluding all 100 previous problems and frozen-prefix hashes; 8 continuations each, cap 1 only, total 800. Sampling seed 2026091711; continuation seeds 2026091702–2026091709. Same Qwen3-1.7B student and Qwen3-14B teacher, k=1, frozen prefix and forced initial teacher token with original entry token banned. After that token, student generates to EOS or original token limit with no more teacher calls.

One node, 4 H800 GPUs, 32 CPUs, 256 GiB; two resident student/teacher pairs, 64 concurrent traces, unchanged worker batch size 8. Warm verl-opd environment reused without package changes. UNLIMITED Slurm allocation; no 5-hour dispatcher cutoff or routine resubmission. Excludes gnho009 and gnho018. All four workers run seeded CUDA witnesses; two pair service tests and a correct/incorrect CRUXEval-I scorer smoke must pass before generation. Four existing branch semantic tests pass locally and are repeated remotely.

On successful generation, the GPU wrapper submits a single 16-CPU scoring task with an afterok dependency, using the unchanged benchmark harness in bwrap without external API calls. Strict prefix, seed, cap and 800 unique score-key coverage checks gate completion markers. This run estimates cap-1 accuracy on a disjoint cohort; it has no new cap-0 paired control.

Runtime: /storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/cruxeval_i_fresh100_cap1_20260917
Outputs: /storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_prefix/cruxeval_i_fresh100_cap1_20260917
Launch: /soft/slurm/bin/sbatch --parsable <runtime>/gpu.slurm after prepare.py succeeds and deployed hashes match.
