# Automatic full greedy baseline scoring

User requested automatic scoring after generation. Submitted CPU job107953, afterany107946,32CPU64GiB,no GPUs,2h. Entry guard requires ALL_GENERATION_COMPLETE.json, so interrupted/incomplete generation cannot be mistaken for complete scoring. afterany permits grading complete generations if the GPU job times out during its built-in code-only scoring stage. Reuses completed code grades with byte-identical scoring signature; no duplicate external grading batch exists.

All9142 outputs (4634code +4508math/reasoning) aggregated across8datasets and2roles. Code uses existing execution harnesses. Math/reasoning uses identical extraction/system prompt/OpenLux gpt-4o-mini,temperature0,max_tokens512,32adaptive concurrency. Three API calibration probes passed through dedicated authenticated TLS relay. API credentials remain on server; token mode600. User authorization recorded under output judge/EXTERNAL_SCORING_APPROVED.json. Timeout generation remains count-wrong; API errors are retried and never silently counted wrong. Full summary.json/csv and ALL_SCORING_COMPLETE.json written only with complete validated coverage. Dedicated relay stops after success.

Output root: /storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_baselines/full_teacher_student_greedy_20260920
