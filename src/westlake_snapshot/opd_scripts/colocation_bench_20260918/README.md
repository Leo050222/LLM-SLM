# H800 14B+1.7B colocation benchmark

One Slurm job,2H800,32CPUs,192GiB,UNLIMITED. Does not alter or resume main experiments. Warm verl-opd reused; existing core and FIFO RPC. No external scoring.

Workload:18 math/reasoning questions:2 random problems in each source-length tercile for each of MATH500,OlympiadBench,BBH. Same original prompts and deterministic seed2026091801 across all phases; k1/8/32,threshold2,unlimited interventions,64-token drafts,original40264/40960 limits. Total54 rollouts perphase,216 across4phases. Historical length stratification is a proxy for generated length, not an enforced truncation. Three-hour infrastructure lease failure is reported, never counted as an incorrect answer or a completed throughput sample.

Phases:split8(current batch8),split4(control batch4),colocated4(two pairs on two GPUs,batch4perengine),split8_repeat(drift check). Split layouts have1pair32traces;colocated has2pairs64traces. All BF16,no quantization,no MPS,no prefix cache. Colocated explicit KV budgets:teacher18GiB,student10GiB; GPU utilization startup flags .65/.23; actual allocator behavior and long-context capacity must be verified in-job. Engines start sequentially and receive independent CUDA witnesses. Service warmup and startup excluded from measured rollout interval but startup cost recorded separately. Each phase fully shuts its own workers down before next phase.

Artifacts in opd_profile/colocation_bench_20260918/runs/<job>:perphase traces, completed rollout latencies/token counts/hashes and RPC queue/batch timing; GPU CSV sampled every2seconds; full vLLM logs retain preemption warnings;SUMMARY.json,RESULTS.json,COMPARISON.json. Compare colocation/current, colocation/batch4control, repeated-baseline drift. Speedups are selected-workload deployment comparisons, not accuracy equivalence or guaranteed whole-grid gains.

Instrumentation files:worker.py adds explicitKV argument and enables vLLM stats;bench.py wrapsRPC and timesfullrollouts;launch.py orchestrates phases and samples nvidia-smi;prepare.py freezesselection;gpu.slurm allocates2GPUs. Original experiment files are unchanged.

Synthetic context capacity probes:8192/16384/32768/40896 tokens,8 output tokens,each role/pair,sequential calls. Recorded separately in CONTEXT_STRESS.json,excluded from rollout speed;does not certify all-long concurrent batch capacity. Selected real response lengths7–2091 tokens,so stress probes supplement the typical workload.
