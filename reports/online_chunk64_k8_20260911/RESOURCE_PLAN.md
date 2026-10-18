# Online test1000 resource plan

User resource ceiling: at most two nodes / sixteen H800 GPUs. There is no
per-question takeover-count limit; the reported maximum is only an observation.

Decision: 2 x H800 (one Qwen3-1.7B student, one Qwen3-14B teacher),
32 allocated CPU threads and 240 GiB host RAM. GPU workers batch up to eight
requests each. Networked coordinator uses 64 concurrent question state machines;
external GPT-5.6-Luna judge does not use these GPUs. Formal Slurm time limit:
24 hours. Rough planning estimate: several hours (about 3–6 if judge latency
and candidate density remain similar); not a measured full-run ETA.

Evidence from the 2026-09-11 smoke:

- Warm existing verl-opd stack: PyTorch 2.11.0, vLLM 0.23.0,
  Transformers 5.6.0. No package installation/rebuild.
- Job 102139 on gnho033: both seeded CUDA witnesses passed. Resident GPU
  memory including the preallocated KV cache: 72,277 and 73,595 MiB, below
  the 81,559 MiB device capacity.
- Fresh independent agent executed VALIDATE_SERVICE.md verbatim and received
  LIVE_WORKER_CONTRACT_OK; student and teacher logprob alignment error 0.0,
  exact 8-token bridge, first-token exclusion, two synthetic handoffs.
- Synthetic long-prefix scoring benchmark, isolated single-request engine time:

| Prefix tokens | Student seconds | Teacher seconds |
|---:|---:|---:|
| 1,024 | 0.102 | 0.370 |
| 8,192 | 0.767 | 1.089 |
| 16,384 | 2.094 | 2.797 |
| 32,768 | 3.869 | 5.531 |

Long contexts above the recorded baseline maximum use repeated baseline token
IDs only for a throughput witness, never for evaluation or anchor quality.

- Existing 1000-question native non-thinking baseline lengths: mean 1303.87,
  median 1114, p90 2212, p99 3991, max 7728 tokens.
- Real judge response latency ranged upward during the smoke: initial probe
  9.35 s; cumulative mean about 26.9 s at 86 calls. At 437 s wall time the
  workers had consumed only 22.6/17.5 batch-seconds (including the independent
  and long-context probes). More GPU replicas would mostly increase idle time.
- Increase question concurrency to hide judge latency; do not increase model
  tensor parallelism because each model already fits one GPU. Real provider
  latency/rate limits can change the duration. Failures remain explicit and
  resumable, never treated as a negative anchor label.

Protocol: all 1000 questions, one rollout each, student chunks 64, strict
|log pS - log pT| > 2, teacher bridge 8, unlimited repeated takeovers inside the
40264-response-token and 40960-model-token limits. Natural EOS stops. Teacher
first token must differ from the rejected student token. Temperature/top_p=1/1.

Input test parquet SHA256:
6ef8e66d181b7522bad2904ad1fcf20d8fc152af0c9d7eb05e07a9f2f0415ae0

Submission is gated on all eight capped real-judge smoke trajectories completing
without failure and the saved token-source/anchor/bridge invariants passing.
These capped trajectories provide functional evidence, not an accuracy claim.
A dependent 4-CPU/16-GiB job runs the existing boolean-normalized scorer on the
complete 1000-question cohort after GPU generation ends.

## Completed validation

Eight real-judge smoke questions completed without failure in 943.6 seconds.
They exercised 44 takeovers and 352 teacher tokens, including ten takeovers in
one question, with 179 committed judge calls. The saved trace/bridge/source-mask
audit returned REAL_JUDGE_SMOKE_AUDIT_OK. Scoring interface also completed.
Seven trajectories hit the diagnostic 1024-token cap, so their accuracy is not a
full-answer result and must not be used as a method-quality estimate.

Smoke Slurm 102139 and both model steps completed 0:0; the node returned idle
before formal submission. All 1000 prompt token sequences, problem IDs and
ground-truth answers were verified equal to the previous native non-thinking
baseline. Eleven new unit tests and seven existing anchor tests passed locally.
