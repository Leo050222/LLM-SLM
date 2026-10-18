# Matched student and teacher greedy baselines

Adds two controls to quartile_support_k_20260918, using byte-identical frozen prompts, sampled source rollouts, quartile assignments, and intervention prefixes:

- k0: Qwen3-1.7B at temperature=0 from the fixed prefix to EOS/length cap, with no new teacher tokens.
- teacher_greedy: Qwen3-14B at temperature=0 from the same fixed prefix to EOS/length cap, with no student continuation.

These are prefix-conditioned continuations; previously generated source-prefix tokens remain intact. Each control has 800 samples (100 per dataset, 25 per quartile). Existing five k branches and their one-time 32-token teacher bridges remain unchanged. New total is 5600 completions across seven groups.

A single-node 8-H800 job starts after original generation job 106797. It sequentially runs eight student replicas, then eight teacher replicas, retaining the allocation and never concurrently allocating a second baseline node. Models, tokenizer, 256-token chunking, greedy temperature, EOS handling, and response/context limits match the original experiment. No new teacher bridges are sampled. GPU kernel and greedy service witnesses execute before each stage.

CPU scoring waits for this GPU job and original scoring job 106798. It uses the identical original grading code/signature and then validates all 5600 input identities, prefix hashes, output hashes and grading signatures. summary_with_baselines.json/csv reports seven groups by dataset and quartile; paired_changes_vs_k0.csv reports changes against student greedy. Original five-group summaries are preserved.

Tests: model role and prefix fidelity, temperature=0, resumed token continuity, incompatible checkpoint rejection, complete seven-way paired aggregation and incomplete coverage rejection.

Remote output: /storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_prefix/quartile_support_k_20260918
Remote code: /storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/quartile_greedy_baselines_20260918
Receipt: BASELINES_LAUNCH.json under the output root.

Submitted generation job 106799 (afterok:106797) and scoring/aggregate job 106800 (afterok:106799:106798). Both verified PENDING on their dependencies. Three local and remote semantic tests passed; all three grading implementation files compare byte-identical to original experiment.
