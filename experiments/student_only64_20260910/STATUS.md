# Student-only 64 rollouts — 2026-09-10

User authorized reusing the existing eight samples per question and generating only 56 additional samples. Three H800 nodes were initially requested; user changed to two if three were not idle. Only gnho033 and gnho034 were idle, so this run uses those two nodes (16 H800 total).

- Generation job: 102064; started 2026-09-10 19:58:21 China time; 2 nodes × 8 H800, 64 CPUs/node, 512 GiB/node, 12h limit.
- Scoring job: 102065; afterok:102064; 3 CPU nodes × 96 allocated CPUs, 80 workers/node, no GPUs.
- Remote code: /storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/student_only64_20260910
- Remote output: /storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_deepmath/student_only64_20260910
- Logs: /storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/s64.102064.gpu{0..15}.log and s64_score.102065.rank{0..2}.{log,err}

Same Qwen3-1.7B model, tokenizer, exact prompt tokens, nonthinking mode, temperature=1, top_p=1, top_k=-1, max_tokens=40264, max_model_len=40960, n=8 as the original baseline. Seven new independent RNG seeds 20260910..20260916 generate groups of eight per question, numbered 8..63; original indices 0..7 retain their existing outputs and scores. All 1000 questions retain their original identity. Old baseline correctness: 2247/8000.

Preparation checks test parquet SHA256, old shard sampling configs and output hashes, uniqueness of old (dataset_index, replicate_index), and snapshots all old scored samples. Each new bundle is written atomically with eight samples; restarting can reuse completed bundles. Scoring uses the same score_math_response implementation and 20-second symbolic timeout, records each source hash, and reuses score caches. Final merge verifies per-question coverage of all 64 indices and source hashes.

Local Python and shell syntax checks passed. Preparation succeeded, and all 16 workers passed seeded CUDA kernel checks. At the initial check, engines were starting; no new bundles had been written yet. Completion has not been claimed. Results will be in summary.json and per_question.json; COMPLETE.json is written only after full generation/scoring/merge.

## Completion
Both jobs COMPLETED 0:0. Final merge finished 2026-09-10 21:07:24 +08:00. Generation 32m29s; scoring/merge 36m04s. All 64000 samples accounted for. Correct=17914/64000 (27.990625%); pass@64=626/1000 (62.6%); all64=21/1000. Score errors=0, length stops=6. Summary and per-question results downloaded and aggregate counts verified locally.
