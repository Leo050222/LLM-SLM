# Four takeover ablations, 2026-09-11
GPU102611 single node8H80064CPU512GiB24h; score102612 afterok10261164CPU48workers192GiB6h. Coordinator3346126 login01. Initial state PENDING Priority; no model readiness claimed.
Remote ROOT=/storage/yukaichengLab/wangxin/test_eiloeo. Run ROOT/opd_eval_hard/takeover_ablation_20260911. Scripts ROOT/opd_scripts/takeover_ablation_20260911.
SSH wangxin@172.16.78.35 port10024. Python /storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python. Frozen original worker code reused from opd_online_code/20260911_chunk64_k8_2nodes.
ONLY MATH500500, OlympiadBench OE_TO_maths_en_COMP674, BBH1080. NO AIME. Same2254 prompts and original dataset_index with gaps from teacher1 experiment. Four modes sequentially share4 worker pairs128 concurrency:
1 luna_cap8: strict absolute gap>2, Luna selects, teacher8tokens, max8 interventions, first-token ban retained.
2 nojudge_cap8: strict absolute gap>2, first chronological candidate, noLuna, teacher8,max8, first-token ban retained.
3 luna_allow: gap>2+Luna, teacher8, unlimited interventions, allow original student token.
4 random_allow: per eligible student token Bernoulli trigger, no threshold scoring/noLuna, teacher8, unlimited interventions, allow original token. User explicitly accepted matching previous per-dataset Luna mean interval. Probability = previous_takeovers/(previous_student_tokens+previous_takeovers), so expected nontrigger student tokens per takeover equals previous student_tokens/takeovers. Rates frozen in random_rates.json using previous whole-dataset Luna8 nonthinking traces; not per-question answers. Math p.002595125777096 (384.338 student tokens), Olympiad .003054130416762 (326.425), BBH .011493698869588 (86.0042). This matches expected frequency, NOT exact teacher budgets.
All nonthinking,temp1,top_p1,seed20260911,student draft64,teacher8,cap40264/context40960. Cancellation of ban preserves teacher1+7 generation split and seeds to avoid sampling-call segmentation confound. Same token allowed, not forced. Random positions chosen chronologically within draft from independently hashed random seed perquestion/round. Stop EOS; do not impose takeover cap in allow groups. Record selected_candidate and student_entry_token to audit repeats. Capped groups stop scoring/judging after8 takeovers but continue student64 drafts.
Original_run.txt and original_core.txt frozen source witnesses. Local+remote4 protocol tests passed: cap8/nojudge zero calls, unlimited and original-token equality, random no score/judge, deterministic checkpoint resume, probability0/1. Syntax checks passed. Manifest hashes all scripts/*.py; NEVER add or edit remote PY after generation starts. Changes need new dir or controlled compatible recovery; audits via stdin.
Coordinator waits8 models then runs four modes with up to3 resumable attempts each, verifies pergroup, after all9016 outputs verified writes ALL_GENERATION_COMPLETE rows9016, workersstop, CPUscore. Score functions unchanged; original IDs used via prompts; group dirs luna_cap8/nojudge_cap8/luna_allow/random_allow each completed,traces,manifest. scores/mode and summary.json.
Total thread GPU<=2nodes16cards including teacher1 GPU102597. Previous threshold3 GPU102590 and score102591 confirmed COMPLETED0:0; results still need audit/sync/report via existing heartbeat. No automatic discards, no restart on observation timeout or normal long traces. Repair only confirmed failures and preserve checkpoints; update score dependency. Finish audit all9016 outputs/scores,prompt identity,max8/ban flags/bridge<=8/judge rules/strictgap/time gate/hash; sync artifacts no large queues, compare same three datasets priorstudent/teacher/Luna8/noLuna8/randommax8 with budget differences disclosed. Report accuracy,length,takeovers,teacher share and equal-original-token counts. Pause heartbeat only after all tracked experiments collected/reported.

CORRECTION BEFORE ANY GENERATION: User intended signed gap student_logprob - teacher_logprob > +2, not absolute. GPU102611 held while still PENDING with zero models/outputs, code updated to filter strict signed positive before Luna/nojudge selection. Manifest settings gap_rule explicitly records corrected protocol. Random group unchanged. Negative and exactly+2 boundary rejection test added. Requested GPU walltime lowered to6h to improve backfill; any interrupted work remains resumable. Prior result experiments remain absolute-gap and must be labeled as such.

After shortening TimeLimit to6h and release, GPU102611 confirmed RUNNING gnho008 (8H800); initial runtime6seconds, models not yet verified ready. CPU102612 dependency unchanged. Fifth protocol test for negative/boundary signed gap passed. Teacher1 GPU102597 continues historical absolute-gap protocol; do not mislabel it signed.

Monitor2026-09-11 18:07:29 luna_cap8 73/2254; other3 awaiting sequential run;8 models ready; manifest confirmed signed student_minus_teacher_gt_positive_threshold; no failures.

Monitor2026-09-11 18:14 luna_cap8 442/2254; other3 notstarted;102611RUNNING12m58; coordinator alive; scores0; noerrors.

Monitor2026-09-11 18:20 luna_cap8 727/2254;remaining3 await;GPU102611RUNNING18m57;coordinator alive;scores0;no errors;normal progress.

Monitor2026-09-11 18:29 ISSUE: luna_cap8 977/2254 at18:28;>=1222 question RuntimeErrors in current pass. GPU jobs and coordinators alive, FAILEDfalse; all GPU logs no error. Two Luna diagnostic requests succeeded (minimal, then exact same config reasoning/extra_body). Selected failed question cached draft+score requests had no worker error. Likely transient/provider/concurrency issue but exact HTTP cause NOT captured by production runner (only RuntimeError). Do not claim confirmed cause. Existing coordinator retries up to3 passes after current futures settle. Next heartbeat urgently inspect run_status/retry counts/live progress. If persistent failures, diagnose actual failing judge request safely (HTTP code/category only), preserve protocol/checkpoints. No scores yet.

18:37 confirmed actual failed candidate Luna request HTTP403 error_type=new_api_error code=local:insufficient_quota. Minimal probes succeeded but full request failed; quota limitation confirmed. Proposed SIGSTOP coordinator process groups2683003+3346126 and run nojudge_cap8 on existing teacher1 queue, random_allow on ablation queue. AUTOMATIC APPROVAL REVIEW REJECTED combined action due lack explicit permission to pause/reuse and possible interference/data races. NO MUTATIONS FROM THAT CALL EXECUTED; original coordinators not paused by us, fallback runs NOT launched. Do NOT bypass or retry this action without user explicit approval. Ask user authorization to pause Luna and reuse two nodes for already-authorized noLuna groups. Meanwhile read-only monitoring is allowed. Need quota restored for Luna.

Monitor18:42 read-only: completed979 unchanged; jobsRUNNING; FAILEDfalse;scores0. Await user approval/quota restoration; no mutations or API probes.

18:44 user explicitly confirmed quota replenished and requested restart. Approved SIGTERM of verified old child run.py processes2757457/3615622; existing parent coordinators automatically retry original config/checkpoints. GPU jobs102597/102611 and score dependencies unchanged. No fallback/queue reuse. teacher1 newchild391079, completed increased1993, FAILEDfalse. Previous quota approval-wait is superseded by user resume instruction.

Monitor18:50 luna_cap8 1428/2254;other3 await;102611RUNNING49m31;FAILEDfalse;scores0;recentlogs show successful completions;normal post-quota-recovery progress.

Monitor18:56 luna_cap8 1751/2254;othersawait;102611RUNNING55m23;FAILEDfalse;scores0;normal progress.

Monitor19:02 luna_cap8 2154/2254;othersawait;GPU102611RUNNING1h01m31;FAILEDfalse;scores0;successful recent completions, normal progress.

Monitor19:08 luna_cap8 2248/2254;remaining3groups awaiting sequential run;GPU jobsRUNNING;FAILEDfalse;scores0;normal progress.

Monitor19:14 luna_cap8 2253/2254;other3await;102611RUNNING1h13m27;FAILEDfalse;scores0;normal tail progress.

Monitor19:20 luna_cap8 2253/2254;remaining179129370tokens age5s;othersawait;all checkpoints advancing;GPU jobsRUNNING;FAILEDfalse;scores0.

Monitor19:32 luna_cap8 2253/2254;remaining179136794tokens age2s;othersawait;live checkpoints advancing;GPU jobsRUNNING;FAILEDfalse;scores0.

Monitor19:38 luna_cap8 2253/2254;othersawait;102611RUNNING1h37m22;FAILEDfalse;scores0;no new actionable state.

Monitor19:45 luna_cap8 COMPLETE2254/2254;nojudge_cap8 started809/2254;luna_allow/random_allow pending;GPU102611RUNNING1h44m16;FAILEDfalse;scores0.

Monitor19:51 luna_cap8 complete2254;nojudge_cap8 2252/2254;allowgroupsawait;GPU102611RUNNING1h50m30;FAILEDfalse;scores0;live progress.

Monitor19:58 luna_cap8 and nojudge_cap8 both COMPLETE2254each;thirdgroup luna_allow started56/2254;randomawait;GPU102611RUNNING1h57m04;FAILEDfalse;scores0.
