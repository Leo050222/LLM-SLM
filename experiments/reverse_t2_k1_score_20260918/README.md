# Reverse experiment scoring

Code-only Slurm job: 106689. Scores 2,317 outputs across five code datasets with existing isolated official harnesses. 32 CPUs, 64 GiB, no GPU. Per-record source hashes and scoring signatures are retained. Both answer correctness and generation-timeout-as-wrong correctness are saved. Code results are stored under the source group's scores directory, with code_summary.json and ALL_CODE_SCORING_COMPLETE.json at the experiment root.

Math/BBH (2,254 rows) scorer is prepared using existing extraction and GPT-4o-mini answer comparison through api.openlux.ai. External submission was rejected by automatic approval review because explicit authorization to send answers to that specific provider is required. No relay or external scoring task was launched. User subsequently explicitly authorized external scoring. After approval, avoid concurrent runs against the shared scoring.lock; the full score script resumes valid existing code scores.

Remote scripts: /storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/reverse_t2_k1_score_20260918
Remote results: /storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_ablation/reverse_t2_k1_20260918

## External scoring authorized and launched

User explicitly approved sending extracted candidate and reference answers to api.openlux.ai using GPT-4o-mini. Code job 106689 completed successfully for all 2,317 rows. Full/resume scorer job **106690** submitted; reuses code scores and evaluates 2,254 math/BBH outputs. Authenticated destination-restricted relay started on login01 and stops when all scoring completes.
