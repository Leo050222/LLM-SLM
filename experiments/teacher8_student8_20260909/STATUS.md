# Teacher 8 × Student 8 held-out handoff

User authorization: Teacher Qwen3-14B from question start, 1000 held-out questions × 8; each teacher trace prefix h32/h64/h128/h256 followed by Student Qwen3-1.7B × 8. Explicit GPU cap: at most 8 H800 on ONE node. GPU stages are sequential in the same full allocation. No recurring monitor is authorized or installed.

Protocol: frozen test SHA 6ef8e66d181b7522bad2904ad1fcf20d8fc152af0c9d7eb05e07a9f2f0415ae0; saved original non-thinking Qwen3-1.7B question prompt IDs checked against current native template. Seed 20260811, temperature/top_p/top_k 1/1/-1, n8 in both stages. Response budget 40264 including teacher prefix, context limit40960. No decode/re-encode at handoff. Naturally finished teacher traces before cutoff are carried through with explicit no-student-generation labels (8 bookkeeping rows; not 8 independent samples); summary separately reports active student prefixes.

Canonical root: /storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_deepmath/teacher8_student8_20260909
Smoke root: canonical root + _smoke (1 question, 8 teacher answers, 256 student bookkeeping/generated outputs).
Deployed scripts: /storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/teacher8_student8_20260909
SSH: wangxin@172.16.78.35 port10024; scheduler commands through bash -ilc; /soft/slurm/bin.

- 101464: smoke GPU, one H800 on gnho032, RUNNING; Teacher weights loading.
- 101465: smoke CPU scoring, afterok101464.
- 101466: FULL GPU, afterok101465, single node 8 H800, 64 CPUs,640G,24h. Teacher then Student in same allocation. No duplicate submissions.
- Full CPU scoring submitted afterok101466 (job ID recorded below once returned).

Files saved atomically per question (teacher,8 outputs) and per question/teacher-replicate/h (student,8 outputs). Exact coverage checks and bundle SHA digest after each generation stage. Scoring checkpoints per bundle, final summary.json, per_question.json, SCORING_COMPLETE.json. GPU logs opd_logs/t8s8.JOBID.{log,err} and t8s8.JOBID.STAGE.RANK.log; CPU logs t8s8_score.JOBID.{log,err}.

Evaluation: teacher sample accuracy/pass@8/all8; student conditional-prefix sample accuracy/pass@8/all8; per-question coverage across64 outputs reported separately; no treating all64 as independent teacher-prefix samples. Compare to previous 1×1 h32/h64/h128/h256 accuracies29.3/28.3/29.3/27.6 and Student-only28.3, with sampling-count differences explicit.

Full CPU job ID: 101467, afterok:101466. Scheduler verification of full GPU: NumNodes=1-1, gres/gpu:H800:8, 64 CPUs,640G. Entire chain submitted with no overlapping GPU stages.

2026-09-09 23:06 CST check: smoke GPU101464 COMPLETED0:0 (6m08s), smoke CPU101465 COMPLETED0:0 (2m29s). Full101466 FAILED1:0 after27s on gnho009, all workers failed CUDA witness: NVIDIA driver version12050 too old for current PyTorch. No full generation occurred. Cancelled impossible dependent CPU101467. Verified gnho032 IDLE with8 H800 and prior successful smoke; recovery full101470 submitted with --nodelist=gnho032, same immutable protocol/scripts and output root, no configuration change. Future GPU submissions for this environment must avoid gnho009 unless separately revalidated.

2026-09-10 12:38 reconnect: 101470 FAILED1:0 at02:00:13 after2h53m20s. Teacher complete8000; saved Student bundles h32/h64/h128/h256=4627/4608/4584/4570 (147112 bookkeeping/generated outputs,57.465625%). Worker logs all end at normal PROGRESS with no traceback; parent has no student exit reports. Slurm ReasonNone, no explicit OOM/timeout; cause unresolved. Storage49T available. gnho032 now occupied by another user's8GPU allocation. Cancelled our impossible CPU101471. Recovery101704 submitted --nodelist=gnho032 using resume.slurm with ERR/TERM/EXIT diagnostics and setsid --wait. Same unmodified pipeline.py/config; cached bundles validated/reused. Only missing student bundles generated. Other project jobs untouched.

2026-09-10 resource correction: 101704 was unnecessarily pinned to busy gnho032 while gnho033/034/035 were IDLE (8 H800 each). Updated existing pending job via scontrol update JobId=101704 ReqNodeList=gnho035, preserving CPU dependency101705. No duplicate submission; requested one node8H800 unchanged.

User authorized CPU parallelization over <=3 nodes, then explicitly capped each node at96 allocated CPUs. Old scoring101705 cancelled with caches preserved. Interrupted initial submission created101758 (3x128CPU); it was cancelled immediately on detection. Revised101760 submitted afterany101705:101758: exactly3 nodes,1 task/node,96CPUs/task,512G/node,80 scoring workers/node (240 total); no GPUs. Unchanged score_file and original scorer reused; shards disjoint by dataset_index%3, cached source hashes checked; bounded2xworkers future queue. srun waits all shards, verifies33000 bundle total, then original score.py cache-only pass/merge. Scripts cpu3.slurm and score_shard.py. Logs t8s8_score3.101760.rank{0,1,2}.{log,err}. Existing generated264000 teacher+student rows unchanged.

2026-09-10 16:36: scoring101760 FAILED1:0 at15:49:42 (15m15s), worker and batch stderr empty, no BATCH_ERROR trap output. srun .0 MaxRSS~67GiB, below512GiB/node; no explicit scheduler OOM/time limit. Cause unresolved; no claim of fix. Cached counts teacher101,h32=800,h64=795,h128=791,h256=783 ->3270 bundles/26160 rows. No active duplicate score job; storage49T free. Recovery101795 submitted using cpu3_resume.slurm with batch EXIT/HUP/TERM traps, per-rank wrapper EXIT reports and verbose srun; same3nodes96CPU/node80workers, original scoring semantics/cache reuse. Generated data untouched.

FINAL 2026-09-10 19:12:59: 101795 and both srun steps COMPLETED0:0, elapsed2h35m08s. All33000 scored bundles/264000 bookkeeping rows present; SCORING_COMPLETE questions1000. Teacher3399/8000=42.4875%,pass8=61.5%,all8=19.6%. Student correct counts h32/64/128/256=18109/18215/18292/19051 each64000; accuracies28.2953/28.4609/28.5813/29.7672%. Conditional-prefix pass8=51.2875/51.1875/51.05/51.3625%,all8=7.775/7.8/8.3/10.1875%. h256 has5 finished teacher prefixes (40 bookkeeping rows); active-only19011/63960=29.7233%. Score_errors0,think_tags0,length_stops12 across all teacher/student rows. Local summary.json/per_question.json/SCORING_COMPLETE.json retrieved. No active monitor to remove.
