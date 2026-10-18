# Two full H800 nodes: online test1000

At the user's request, the previous two-GPU execution was stopped and replaced
by **two exclusive nodes, sixteen H800 GPUs**:

- GPU job **102155**, nodes **gnho011 + gnho034**, 8 H800s per node.
- **8 independent student/teacher pairs**: 8 student cards and 8 teacher cards.
- **256 concurrent questions**, deterministically assigned by dataset index modulo8.
- Dependent scoring job **102156**, released after GPU generation finishes.
- Coordinator PID **2537056** on login02; recovery guardian PID **2599105**.

There is **no per-question takeover-count limit**. Student chunks64, teacher
bridges8, strict absolute logprob gap>2, judge configuration, seeds, input set,
response/context limits and scoring rule remain the same. The entire fixed
1000-question cohort is still evaluated once per question.

Migration retained **72 completed responses**, **134 saved trace checkpoints**,
**2553 cached judge decisions**, and completed GPU RPC caches. Files were linked
into a new run directory after the old coordinator and GPU workers stopped;
atomic replacement prevents subsequent writes from changing source artifacts.
The new coordinator validates scientific protocol equality against the saved
source manifest before resuming. Unfinished work resumes from its last committed
checkpoint; completed answers are retained. Old GPU job102149 ended0:0, and its
obsolete scoring dependency102150 was cancelled. No third GPU node overlaps.

Remote directory:
`/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_deepmath/online_chunk64_k8_test1000_2nodes_seed20260911`

New immutable code snapshot:
`/storage/yukaichengLab/wangxin/test_eiloeo/opd_online_code/20260911_chunk64_k8_2nodes`

`results/migration.json` records provenance; `launch.json` records resource and
concurrency settings. Each queue/replica_N contains its own resident workers.
`results/completed`, `results/traces`, `results/summary.json` and final completion
markers retain the earlier layout. The external judge can still limit throughput;
no eightfold speedup is claimed before measurement.

Physical binding verification passed: each EngineCore PID maps to a distinct
GPU UUID, eight per node. Slurm process-local CUDA indices are all remapped to0,
so those local indices must not be treated as physical GPU identifiers. A fresh
independent agent executed VERIFY.md verbatim and received
TWO_NODE_16GPU_VERIFIED, confirming16 H800 witnesses,8 replicas,256 concurrency,
and preservation of72 completed responses and134 checkpoints.

The original shell supervisor2527000 was replaced without interrupting the
coordinator or GPU workers. A transient external-judge RemoteDisconnected on
question321 is retried after the current pass by rerunning the identical frozen
command, at most three recovery passes. Completed answers are skipped and
unfinished traces resume. This retry count does not limit teacher takeovers.
The guardian releases workers after success or exhausted recovery, and only
writes GENERATION_COMPLETE after all1000 answers finish. At the final launch
check, GPU job102155 remained RUNNING on both nodes, no STOP markers existed,
and261 completed answers were present. Full-cohort accuracy remains pending.
