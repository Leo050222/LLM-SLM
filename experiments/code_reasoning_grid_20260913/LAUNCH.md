# Code reasoning evaluation — 2026-09-13

User confirmed CRUXEval-O/I, HumanEval+, MBPP+, LiveCodeBench; exclude BigCodeBench. GPU ceiling: **four H800s total**.

Job **103184**, RUNNING on **gnho020**, 4 H800 / 32 CPU / 256 GiB. Four resident engines: two Qwen3-1.7B / Qwen3-14B pairs. First launch witness: four model identities ready, actual completed outputs, no dispatcher failure. Snapshot: student-only 2294/2317, teacher-only 2022/2317.

| Dataset | Fixed version | Questions |
|---|---|---:|
| CRUXEval-O | repo 190faf16d175b5847b0af05d937872b1fb395942 | 800 |
| CRUXEval-I | same corpus | 800 |
| HumanEval+ | v0.1.10 | 164 |
| MBPP+ | v0.2.0 | 378 |
| LiveCodeBench | cached incremental v6, 2025-01-04–2025-04-06; **not cumulative release_v6** | 175 |

2317 questions × 27 configurations = **62559 rollouts**. Two baselines plus k=1,4,8,16,32 × signed threshold=2,2.5,3,3.5,4. No Luna; unlimited intervention count; teacher entry token bans the replaced student token, matching the preceding grid. Positive signed logprob(student)-logprob(teacher) must be strictly greater than threshold.

All configurations: native non-thinking, temperature=1, top_p=1, seed=20260911, response cap40264/context40960, 64-token student drafts. Baselines use direct unconstrained 64-token generation calls with the same round-seed rule; teacher-only does not split off or constrain its first token. New task-specific zero-shot prompts; do not compare these scores against unrelated prompting protocols without qualification.

Per-trace cumulative active wall-time cutoff900s, includes RPC waiting; excludes unassigned queue/job downtime. Preserve partial output and force timeout incorrect. Shared queue across configs,64 concurrent traces,32 per model pair. A segment stops admitting new traces at5.5h; current traces finish within their remaining15min budget, then a dependent6h job resumes any remaining work. At most one four-GPU allocation runs at once. Checkpoints and cumulative timers persist across segments.

Remote data/results: `/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_code/code_reasoning_grid_20260913`.
Remote scripts: `/storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/code_reasoning_grid_20260913`.
Frozen GPU engine: `opd_online_code/20260911_chunk64_k8_2nodes`.
GPU environment reused without package changes: `miniconda3/envs/verl-opd`.

Scoring uses the pinned official CRUXEval execution checker, EvalPlus base AND plus tests with official input deserialization/special oracles, and LiveCodeBench run_test (6sec/test). Code extraction: first fenced Python block, or full response fallback; CRUX takes the final explicit INPUT/OUTPUT block. It does not repair solutions. Scoring runs in a fresh bubblewrap network/filesystem namespace per trace, exposing readonly interpreter, system libs and evaluation source, with no home directory or inherited credentials. Scorer implementation errors stop the scoring shard rather than being silently counted wrong.

All5 positive/negative scoring witnesses passed on the server. Baseline stop/role/resume/token-budget unit tests and existing deadline-budget/cross-group tests passed locally. Full CPU-node evaluation will only start after the strict ALL_GENERATION_COMPLETE gate.

One-shot finish coordinator PID413297 waits for the generation gate, validates raw data hashes, submits3 CPU shards (at most3nodes,64CPU/node,24 concurrent sandbox runners/node), then writes FINAL_COMPARISON.csv including accuracy, mean interventions, mean output tokens, teacher-token share and timeouts. It exits after completion and does not recreate the deleted Codex UI automation. Track `finish.log`, `score_submission.json`, `job_ids.txt` and `EVALUATION_COMPLETE.json`.

## Temporary single-node8-GPU acceleration

User updated resource authorization: use one node /8 H800 until **2026-09-13 10:00 Asia/Shanghai**, then4 GPUs if unfinished. Submitted8-GPU job103185 on gnho020 with64CPU/512GiB, and4-GPU successor103186 with`afterany:103185`. Cancelled original103184 only after both jobs were recorded. This preserves all committed checkpoints; active budgets resume from the latest periodic save (up to roughly2sec timing granularity).

Separate execution scripts: local`scale8/`, remote`opd_scripts/code_reasoning_grid_scale8_20260913`. No change to scientific settings, model workers, prompt set or scoring. Runtime now supports4 model pairs/128 concurrent traces for8 GPUs and2pairs/64traces for4GPUs; per-pair concurrency remains32.8-GPU wrapper uses an absolute server-time cancellation at10:00; successor was pre-submitted, has a generation-complete no-op guard, and all further continuations request4 GPUs. A one-shot Codex heartbeat `10-4` independently checks the switch at10:00. Scaling submission record: remote`scale_submission.json`.

## 10:00 audit and recovery

The10:00 heartbeat found an earlier startup failure:103185 FAILED at04:46:35, successor103186 FAILED at04:48:33. Both aborted on `('student_only','settings')` during resume manifest validation. Exact discrepancy was persisted `random_rates=[]` versus in-memory `random_rates=()`; all27 scientific setting dictionaries matched after JSON normalization. No8-GPU rollout progress occurred. Previous report had only verified allocation/engine startup, not resumed generation.

Fixed the scale dispatcher by canonicalizing newly constructed manifests through JSON before comparison; no thresholds, seeds, model, input or deadline changes. All27 saved settings checked against expected configurations. Submitted unique4-H800 recovery103198; initial state PENDING(Priority). Existing results5058: student2294, teacher2313, k1_t2451. Finish coordinator413297 remains alive. Recovery record remote`recovery_10am.json`. Extra8-GPU allocation is no longer active or pending.

##15:15 status check / Student unused-top16 failure

Recovery103198 started14:49:44; actual checkpoint progress reached6238 (student2317 and teacher2317 both VERIFIED; k1_t2=1604). Worker failed15:00:56 after Student scoring `sorted_topk` reported12 rather than16 valid candidates. Student top_ids/top_lp are never consumed by the grid Backend, which uses Student chosen logprobs and Teacher top16. Independent scale8/worker.py now retains exact Student chosen logprobs but omits unused Student top16 arrays. Teacher processing and generation unchanged. Old frozen worker untouched. Strict manifest migration accepts only the recorded old/new worker hashes with all other model fields equal and retains pre-migration manifests. Runtime12-candidate regression executed actual patched loop and passed; all27 old hashes matched. Submitted4GPU recovery103234. Completed results retained; final scoring remains gated on all generation.
