# MATH500 three experiments — 2026-09-11

User requested three single-sample nonthinking groups on one 8xH800 node, scoring only after all generation finishes.

Final generation job: 102194 on gnho011 (1 node, 8 H800, 64 CPUs, 512 GiB, 12h limit). Scoring: 102192, afterok:102194, 32 CPUs/24 scoring processes, no GPUs. Coordinator PID 3306395 on login host, networked OpenLux judge using existing secret-file reference. No secret contents copied.

Superseded: 102191 canceled while pending on gnho034, which became occupied. 102193 exited because the startup-failure marker had not yet been cleared. Import path repaired to existing validated deployment: /storage/yukaichengLab/wangxin/test_eiloeo/opd_online_code/20260911_chunk64_k8_2nodes. No generated samples from these attempts.

Remote output: /storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_math500/math500_three_20260911
Remote scripts: /storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/math500_three_20260911

500 frozen MATH500 records, SHA256 35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132. Ground truth from answer (not solution prose). Same prompt as earlier EvalScope MATH500, native enable_thinking=False for all groups. Seed20260911, temperature1/top_p1, max output40264/context40960.

Groups: online (chunk64, gap2, teacher8, unlimited repeated takeovers, original judge and first-token ban); student direct; teacher first256 then student, retaining natural teacher completion if it ends before256. Resident 4 student/teacher pairs. Online completes first, then direct, then prefix256. All 1500 outputs must validate before ALL_GENERATION_COMPLETE.json is written. Only then GPU job exits and scoring dependency releases.

Scoring: EvalScope MATH500 parser.extract_answer(response) followed by math_equal(prediction,strip_answer_string(answer)), matching numeric acc metric. 30-second process alarm per item; exceptions/timeouts explicitly counted. No DeepMath boolean scoring substitution. Original thinking baseline90.4% is a different mode.

Initial syntax checks passed. Prompt preparation completed and all 8 workers have assigned GPU bindings; engine readiness and generation progress must be checked before claiming healthy rollout.

## Complete
102194 COMPLETED 0:0, generation18m32s. 102192 COMPLETED 0:0, scoring10s. All1500 outputs/scoring downloaded and source hashes verified locally. Results: student72.8%, online81.4%, prefix25675.2%. See REPORT.md and audit.json.
