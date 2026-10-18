# Online multi-takeover test1000 launch

Superseded at user request by [two full nodes / 16 H800s](scale_2nodes/LAUNCH.md).
The original results below were migrated, not discarded.

The eight-question end-to-end smoke passed with zero failures: 44 real judge
triggered takeovers, including ten in one question. Independent GPU service
validation found zero student/teacher logprob alignment error; exact bridge and
repeated-handoff checks passed. All 1000 prompt token IDs, problem IDs and targets
match the previous native non-thinking baseline. The smoke used a 1024-token cap
and is not an accuracy comparison.

Formal jobs:

- Generation: **102149**, one node **gnho033**, **2 H800 GPUs**, student and teacher
  each resident on one card, 32 CPUs and 240 GiB RAM, 24-hour time limit.
- Scoring: **102150**, 4 CPUs / 16 GiB, dependent on successful generation job exit.
- Networked coordinator: login02, parent PID **2395045**, **64 concurrent questions**.
- User ceiling: **at most two nodes / sixteen H800 GPUs**. Judge waits dominated
  measurements, so this run uses two GPUs rather than allocating the ceiling.

Protocol: all **1000** fixed test questions, **one rollout per question**, student
chunks **64**, teacher bridge **8**, strict absolute natural-log probability gap
**> 2**, external GPT-5.6-Luna judgment. **No per-question takeover-count limit.**
EOS and the normal response/context limits (40264/40960) terminate generation.
Teacher entry token differs from the rejected student token. Seeds/settings and
model/judge identities are recorded in the immutable run manifest.

Remote run directory:
`/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_deepmath/online_chunk64_k8_test1000_seed20260911`

- `launch.json`: scheduler IDs, resource ceiling and null takeover-count limit.
- `results/traces/`: atomic in-progress checkpoints and takeover events.
- `results/completed/`: final responses and per-token source masks.
- `logs/coordinator.log`: completed/failed question IDs.
- `results/run_status.json`: full cohort completion/failure record.
- `results/summary.json`: final accuracy and intervention-cost totals after scoring.
- `GENERATION_COMPLETE` / `EVALUATION_COMPLETE`: strict completion markers.

The coordinator stops resident workers through queue/STOP when it exits, releasing
GPU resources. Failed questions are explicit and resumable; no partial-cohort
accuracy is reported. Scoring reuses the previous anchor experiment's
boolean-normalized scorer.

See [resource measurements](RESOURCE_PLAN.md), [submission script](submit_formal.sh),
and [saved smoke audit](evidence/smoke_audit.json). Expected duration is several
hours, heavily dependent on external judge latency; full results are not yet available.

Launch verification: job102149 was RUNNING on gnho033; both model.json
readiness records existed and the coordinator had completed question45.
Job102150 was PENDING solely on its generation dependency.
