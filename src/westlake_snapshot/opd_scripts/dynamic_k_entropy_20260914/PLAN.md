# Dynamic teacher support at threshold2

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
