COMPLETE VERIFIED2026-09-15: GPU104045 COMPLETED03:47:08(1h55m58s), CPU104046 COMPLETED03:48:49(68s). Both resources released.4571/4571 generations and scores verified, all source SHA256/IDs/prompts/token accounting valid; original120 completions hash unchanged. Correct2768;90 generation timeouts (78pre-repair exposed,12fresh) and2LCB scoring execution timeouts counted wrong. Local results/ contains individual scores, manifests, summary, FINAL_AUDIT and fixedk1/k8 comparison.csv+RESULTS.md. All final work complete; monitoring should now be PAUSED.

VERIFIED LIVE 2026-09-15 01:56:01: GPU104045 RUNNINGgnho020/4H800; scorer104046 PENDING Dependency. Maintenance COMPLETE. Four ENTROPY_NUMERICAL_WITNESS and SMOKE_PASSED passed; real manifests migrated successfully and formal checkpoint resume progressing.203 completed total:120 preserved+83 newly ended(59stop,24timeout). All24 new timeout timing files existed pre-repair with791.96–862.94sec already consumed; remaining cumulative900sec budget preserved. execution_history.resumed denotes TOKEN checkpoint only:183 had old timing but no saved token checkpoint; use archived timings to identify pre-repair exposure.120 original completed hashes unchanged. Recent160results perworker show mean queue wait0.86–2.69sec, max3.83sec; pending oldest0.35–3.47sec. Student entropy batch0.043–0.054sec, Teacher0.216–0.232sec.12-second GPU means30.1/46.2/40.8/27.0percent, versus prior2.25/6.58/10/20.92 snapshot(not matched workload). Evidence reports/dynamic_k_performance_20260915/final_live_104045.json and gpu_live.json. All11 tests passed; both actual historical manifest migration/re-entry dry-runs passed. Continue monitoring until generation+scoring completion.

LATEST:GPU104045; scorer104046 afterok104045; fixedgnho020/4H800/48CPU. All4 prior104041 numerical witnesses passed(max same-logit old/new difference4.44e-16), but manifest tuple/list mismatch prevented formal generation. JSON canonical comparison now fixed; BOTH real historical manifests passed migration+idempotent resume dry-run before submission.11 tests passed. Old486 artifact hashes unchanged,120 completions preserved.104042 cancelled pending.104045 awaiting live markers; final optimized score path remains original top16.

LATEST:GPU104041; scorer104042 afterok104041; fixedgnho020/4H800/48CPU.104038 preflight passed same-logit entropy equivalence but independent-forward repeat differed>1e-4; validation now independently computes full-vocab CPU reference from SAME logits in BOTH full-return and compact passes, strict1e-7, plus identical-forward control to record variability. Original score16 retained.10 tests passed; all486 old artifacts still hash-identical.104039 pending cancelled. Awaiting4-worker numerical witnesses and formal smoke.

LATEST:GPU104038; scorer104039 afterok104038; fixedgnho020/4H800.104035 preflight scalar entropy passed Student; optional score0 optimization failed equivalence0.113621 and was REVERTED. Final worker preserves original prompt_logprobs16 for BOTH models; only FIFO+leases+GPU entropy transport changes.10 tests passed. All486 old artifact hashes still unchanged,120 completions preserved. Awaiting104038 four-worker witness and formal smoke; do not claim rollout active until markers pass.104036 cancelled while pending.

LATEST:GPU104035,scorer104036 afterok104035; fixedgnho020/4H800.104029 failed only in preflight because vLLM0.23 internal IDs append random8hex suffix. Strict mapping now tested (including out-of-order and ambiguous IDs);10 tests passed remotely. All486 preserved file hashes unchanged after failed preflight, zero formal resume.104030 pending scorer cancelled. Live numerical validation of104035 pending.

LATEST 2026-09-15 performance repair submitted: GPU104029 fixedgnho020/4H800/48CPU; scorer104030 afterok104029/32CPU/0GPU. Previous104008 stopped by verified dispatcher SIGTERM (no scancel on sharednode); other jobs confirmed running. Pending104009 cancelled. Old source/manifests and all486 completed/checkpoint/timing files archived remotely in recovery_104008_performance;120 completions retained (67stop,53timeout). Target source hash explicitly locked; manifest migration verifies every scientific field and model identity except worker code/protocol1->2. Nine tests passed remotely. FIFO oldest-first homogeneous batches, per-waiter deadline leases, orphan cleanup, compact full-vocab GPU entropy with ID/position checks, Student actual-token-only score output. Four real-model numerical witnesses and service smoke MUST pass before formal resume; currently awaiting live validation. Resume cumulative900s and saved tokens; do not re-run old timeouts.

## Historical maintenance entry (resolved by104045)
Current user explicitly requested FIFO, orphan cleanup and GPU-scalar entropy fixes. Maintenance completed; follow the latest verified104045 entry and active monitor. Preserve all completed outputs and cumulative timing. Stop only experiment coordinator PID1043772 inside104008; no scancel on sharedgnho020. Final replacement IDs will be prepended after validation.

# Dynamic teacher support at threshold2

2026-09-15 01:04 monitor:104008 RUNNING ongnho020, four engines initialized.
SMOKE_PASSED.json verifies real full-vocab entropy before/after teacher token on
both model pairs. Formal generation started:57 math checkpoints,0 complete at
snapshot; no FAILED marker. Scorer104009 remains PENDING Dependency.

LATEST2026-09-15 00:43:103977 FAILED in smoke00:02:37, zero generated traces.
Explicit full-vocab count reached vLLM Triton compute_token_logprobs and OOM.
New per-job entropy_extension (worker_extension_cls) replaces only gathers wider
than4096 with FP32torch.log_softmax+gather; small top16 path unchanged. No shared
site-packages edits, no model/policy change. CPU151936-vocab test passed, maximum
error vs float64 reference1.789e-6; full probability normalization passed.
Old worker/manifests archived in recovery_103977_large_k with zero-trace guard.
New generation104008 fixedgnho020/4H800, PENDING Priority; dependent scorer104009
afterok:104008. Old impossible scorer104002 cancelled.
GPU smoke still pending scheduling; do not claim live validation until it passes.

Latest recovery2026-09-15:103943 FAILED during pre-generation entropy smoke,
zero completed/traces confirmed. vLLM -1 full-logprob branch returns raw tensors
with empty token/rank arrays, incompatible with offline output conversion.
worker.py now requests explicit config.vocab_size positive logprob count and
checks exact vocabulary coverage. No policy change; local tests5/5 passed.
Old worker and both manifests archived remotely under recovery_103943_full_vocab.
New GPU job103977, fixedgnho020/4H800, observed RUNNING; new scorer104002
depends onafterok:103977. Old impossible scorer103944 cancelled.

Latest user-requested resubmission: old pending103901/103902 cancelled with
zero runtime. New GPU103943 fixedgnho020/4H800; new scorer103944 depends on
afterok:103943. Experimental code and input settings unchanged.

User-directed node change: while103901 was still PENDING, ReqNodeList updated
fromgnho020 tognho018 on2026-09-14. Job ID and4GPU request retained; scorer103902
still depends on afterok:103901. Node018 was IDLE+PLANNED at the check.

Submitted2026-09-14: GPU job103901 dynamic_k_entropy, fixed gnho020/4xH800;
dependent CPU scorer103902 dynamic_k_score, afterok:103901,32 cores,zero GPUs.
Remote CPU policy tests passed5/5 and4571 prompts validated before submission.
GPU smoke awaits scheduling; no claim of live full-vocabulary service validation yet.

User authorized one experiment, queued ongnho020 with4 H800 GPUs. Scope follows
the8 previous benchmarks,4571 questions,one non-thinking rollout each,no AIME:
MATH500500,OlympiadBench674,BBH1080,CRUXEval-O/I800 each,HumanEval+164,
MBPP+378,LiveCodeBench v6 incremental175. All prompts copied byte-for-byte.

Unchanged gap trigger: earliest candidate in student64-token draft with signed
logprob(student)-logprob(teacher)>2. No Luna. First teacher token bans the replaced
student token. After each retained teacher token, if fewer than8 emitted in this
round and no EOS/budget end, compute full-vocabulary next-token entropy for both
models on the same prefix. Continue iff Hstudent>=0.5 AND Hteacher<=0.2 (inclusive,
nats). Recompute at each step, do not choose k once. On false return to student;
at8 return to student regardless. No per-trace takeover-count limit.

Teacher extension tokens are unconstrained; only bridge entry has first-token
ban. Per-round seed matches historical seed20260911; extension j uses seed+j.
All sampling remains temperature1,top_p1. Sequential one-token sampling is an
intentional generation-organization change versus a fixed-k tail call.

Warm unchanged verl-opd; vLLM0.23 source confirms logprobs=-1 and max_logprobs=-1
request full vocabulary. Entropy operation performs an unmodified one-token
distribution query, discards the diagnostic generated token, computes entropy
from all returned logprobs and checks vocabulary coverage and probability mass.
No top16 entropy approximation. Four worker engines execute a real entropy/
teacher-token smoke before formal rollouts, fail closed on service errors.

Two resident model pairs,64 trace concurrency,48CPU,320GiB,one24h GPU allocation.
Cumulative900sec per trace includes RPC waits, excludes waiting for a queue slot
and job downtime; timeout partial outputs count wrong without automatic retries.
Checkpoint each accepted student prefix and each teacher token. Resume pending
bridge and its saved entropy decisions without replaying accepted tokens.

Meaningful CPU tests passed: inclusive entropy boundaries, either condition false,
max8, first-token-only ban,EOS,partial preservation on timeout and bridge resume.
Immutable output/source manifests and per-event stopping-policy verification.

All4571 generations must verify before dependent32CPU scoring starts (no GPUs).
Use existing math scorers and isolated code sandbox, unchanged answer extraction;
report acc,mean intervention rounds,teacher token share,mean output length,
actual support-length histogram and mean k. Baseline references remain existing
fixed-k1/k8,threshold2 results; stopping/sequential-generation differences noted.
