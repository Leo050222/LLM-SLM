# Teacher intervention entropy diagnostic

Status: submitted on2026-09-14 as Slurm job103661, name intervention_entropy,
explicit --nodelist=gnho041, two H800s and16 CPU cores. Remote frozen sample validated:
4000 traces,3807 with interventions,193 without,25008 measurement records.
Input preparation and local semantic tests passed; see live logs for GPU progress.

Use existing frozen non-thinking Qwen3-1.7B/14B rollouts, T=1, top_p=1,
positive signed threshold, no Luna, first teacher token bans original student token.
Reuse the warm verl-opd environment. No model/data/package downloads or changes.
Request one node, two H800 GPUs, 16 CPU cores, 160 GiB RAM, six-hour bound.
Each model uses one GPU. All results stay below test_eiloeo/opd_analysis.

Sample 20 question IDs per dataset deterministically with seed20260914, without
looking at accuracy/intervention counts/length. Use the same IDs in all25 configs
(threshold2/2.5/3/3.5/4, k1/4/8/16/32). All8 previous datasets included; no AIME.
Each trace contributes its first intervention and independently one uniformly
selected intervention. Report these two estimators separately (never pool them).
No-intervention traces remain in cohort accounting and have undefined post-help
entropy, not zero. EOS-ending bridges are reported/excluded from next-token
comparisons. Early-shortened bridges retain actual length metadata and are excluded
from exact-k post-intervention comparisons.

1. Actual rollout: before teacher entry and immediately after its retained bridge.
2. Matched-prefix control: in k32 traces whose sampled bridge contains32 tokens,
   measure the same anchor at teacher-prefix lengths1/4/8/16/32. Require valid
   nonterminal prefixes for allfive horizons. This tests a common teacher path;
   it is not a new rollout at each k and does not measure downstream accuracy.

Compute H=-sum softmax(raw_logits)*log_softmax(raw_logits), natural-log nats,
over the model's complete output vocabulary at T=1. Both models see exactly the
same token IDs. Apply no top-k/top-p truncation, no first-token ban to measurement
logits. Historical partial logprobs cannot estimate this entropy exactly.
Use BF16 model forward and FP32 log_softmax/entropy. Recompute selected positions
with teacher forcing, no sampling. A prefix ending at token index j predicts j+1;
probe hidden index=len(prompt)+response_offset-1. Validate this against a separate
truncated-prefix forward (0.05-nat numerical tolerance); verify tokenizer identity,
CUDA kernels, entropy bounds, source hashes, config settings, and paired coverage.

Report per dataset/threshold/k: before/after mean entropy for both models,
paired mean changes and entropy gap, fraction H_student>H_teacher, Wilson95%CI,
sample denominators, zero-intervention/EOS exclusions and raw paired samples.
Also paired k-vs1 fraction differences with question bootstrap95%CIs using shared
question IDs. n20 is exploratory; multiple comparisons are not confirmatory.
Actual-k trajectories diverge and have different interventions; matched32 reduces
that confounding but conditions on a32-token bridge and is not causal proof.
Historical stopping policies differ; retained timeout prefixes remain eligible.

Invocation after syncing only this directory:

    PY=/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python
    R=/storage/yukaichengLab/wangxin/test_eiloeo
    $PY $R/opd_scripts/intervention_entropy_20260914/entropy_probe.py prepare --out $R/opd_analysis/intervention_entropy_20260914
    /soft/slurm/bin/sbatch $R/opd_scripts/intervention_entropy_20260914/run.slurm

Verify Slurm RUNNING, CUDA_WITNESS_OK and alignment_witness.json for both models;
then record job ID/node in this file. Completion requires both model COMPLETE
markers and final COMPLETE.json plus CSVs. Scripts resume per-trace checkpoints.
