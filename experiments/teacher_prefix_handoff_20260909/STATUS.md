# Teacher prefix handoff on held-out 1000

Submitted 2026-09-09 via wangxin@172.16.78.35:10024.

- Smoke job: 101447, 8 questions × 4 lengths, 1 H800 on gnho008.
- Full job: 101448, afterok:101447, 1000 questions × 4 lengths, 1 H800.
- Full output: /storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_deepmath/teacher_prefix_handoff_20260909
- Smoke output: same path with suffix _smoke.
- Logs: /storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/teacher_prefix_handoff.JOBID.{log,err}

Frozen teacher Qwen3-14B and original student Qwen3-1.7B: 436/1000 and 283/1000. Existing token IDs and non-thinking question prompts are reused, no decode/re-encode at the handoff. Prefix lengths 32/64/128/256, one sample per condition, seed 20260811, temperature 1, top_p 1, top_k -1. Total response budget (teacher prefix plus student) 40264; context cap 40960. Score the concatenated response with the original baseline scorer. Teacher responses already finished at the requested cutoff are retained and counted separately. Unscored generations and condition-level scored JSON files are saved, with source/script hashes in config.json.

Initial validation: remote teacher summary and completion marker present, exact 1000 rows, 436 correct; all aligned source/prompt assertions passed, CUDA witness passed on initially empty H800. GPU generation startup confirmed. Completion pending. Do not duplicate either job.

Smoke 101447 completed successfully (Slurm COMPLETED 0:0); all four conditions scored 8 unique questions, no scoring timeouts/errors or length stops. Full job 101448 remains the sole full submission.

Background automation was NOT created: automatic approval rejected the proposal because it included future repair/cancel/resubmit permissions. A narrower read-only monitor was requested from the user and awaits explicit authorization.

User requested multiple GPUs and generation first. Serial job 101448 was cancelled; its h32.unscored.json (1000 rollouts) is preserved. Replacement 101455 RUNNING on gnho032: one node, 3 H800, one independent worker for each h64/h128/h256, no GPU-side scoring. Output subdirectories parallel/h64, parallel/h128, parallel/h256. CPU-only scoring 101456 depends on afterok:101455, uses 16 process workers in one 32-CPU allocation and writes canonical h32/h64/h128/h256.json plus summary.json and COMPLETE. Source protocol unchanged; updated per-worker configs record new script hashes. No recurring monitor exists.

Final: CPU scoring 101456 COMPLETED 0:0 in 13m59s; COMPLETE exists. Correct counts h32/h64/h128/h256 = 293/283/293/276 per 1000, versus student-only 283. Rescued/harmed respectively 85/75, 81/81, 89/79, 88/95. All conditions: zero length stops and zero score timeouts/errors. h256 contains one already-finished teacher answer. No monotonic gain observed in this single-rollout evaluation.
