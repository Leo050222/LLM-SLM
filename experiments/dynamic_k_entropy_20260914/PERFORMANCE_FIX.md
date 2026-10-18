# Dynamic-k performance repair, 2026-09-15

Old GPU job104008 was stopped through its verified coordinator PID1043772,
not Slurm cancellation. Other co-tenant jobs ongnho020 stayed RUNNING. Its
120 completed traces (67 stop,53 trace_timeout) and486 output/checkpoint/timing
files were archived with hashes in remote recovery_104008_performance.

Changes:
- Oldest live request selects the next operation batch; hash names no longer
  determine priority. Same-op batching avoids entropy/score waiting inside a
  64-token decode batch. Joining a cached request cannot reset its arrival age.
- Every caller has its own expiry lease. Locks serialize registration, worker
  selection, publication and cleanup. A caller timeout releases only its lease;
  request deletion requires zero live waiters. In-flight computation may still
  finish, but abandoned requests cannot accumulate in the pending queue.
- Explicit entropy-only batches activate a worker extension that reduces raw
  logits on GPU. FP32 log-softmax and FP64 exp/reduction match the old arithmetic;
  only entropy, mass, vocabulary size, request ID and token position return to
  Python. Capture is inactive for normal sampling and scoring. Full-vocab sorting
  and151936 Python logprob objects per entropy request disappear.
- Both models retain their original prompt_logprobs16 scoring path. An optional
  Student actual-token-only optimization was rejected by numerical verification
  (max observed difference0.113621); it is not part of the deployed repair.
- Response metrics include queue_wait_seconds, worker_batch_seconds and
  postprocess_seconds. Per-trace execution_history identifies resumed traces.

Scientific settings, input IDs, prompts, thresholds, entropy conditions,
sampling, first-token ban,900-second cumulative wall budget and checkpoints
are preserved. No automatic retry of existing timeout traces. Older timeouts
remain infrastructure-confounded and must be disclosed in accuracy reports.
Prefix caching remains disabled because an earlier reuse validation did not
pass numerical-equivalence requirements for candidate scores.

Validation command (warm remote verl-opd Python):
python -m unittest discover -s opd_scripts/dynamic_k_entropy_20260914 -p 'test_*.py'
All11 policy/queue/migration/ID-mapping tests passed before submission; actual
historical manifests also passed migration plus idempotent-resume dry runs.
Each of4 workers must pass ENTROPY_NUMERICAL_WITNESS (same-logit scalar versus
old full-vocab return <1e-7; compact path same-logit CPU reference <1e-7).
Independent full/compact and identical compact/compact forward differences are
recorded separately, not conflated with entropy reduction equivalence.
Only then do ready markers permit dispatcher service smoke and formal generation.
Manifest migration accepts only archived source and locked target hashes and
checks every scientific field; subsequent resumes must match migrated manifest.

Latest GPU104045:gnho020/4H800/48CPU; scorer104046:afterok104045/32CPU/0GPU.
Live verification passed on104045: all4 witnesses and service smoke; formal
checkpoint resume is running. See evidence below.

104041 four-model same-logit witness max absolute error4.44e-16. Independent
identical compact-forward control varied up to0.06111 nats for Student; this
confirms forward-repeat variability rather than entropy-reduction error.

## Verified runtime snapshot, 01:56:01

203 completed =120 preserved +83 newly ended (59 normal,24 timeout).
All24 new timeouts had pre-repair timing records with792–863 seconds consumed;
900-second cumulative deadlines were preserved. Token-checkpoint resumed=False
alone does not mean a fresh timing budget; consult archived timing files.

Recent160 results per worker: average queue waits0.86–2.69 seconds, maximum3.83;
oldest pending0.35–3.47 seconds. Student entropy batches0.043–0.054 seconds,
Teacher0.216–0.232 seconds. Both models still score with top16.
12-second GPU utilization means30.1/46.2/40.8/27.0 percent. These are runtime
snapshots, not matched-workload end-to-end speedup measurements.
All120 original completed hashes verified unchanged after formal resume.
Evidence: reports/dynamic_k_performance_20260915/final_live_104045.json,
gpu_live.json. Automation k remains active with latest IDs and preservation rules.
