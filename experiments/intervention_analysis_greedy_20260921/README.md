# Dual-greedy intervention distribution analysis

Scope: all 43,879 interventions from 4,571 dual-greedy k=1, pi=2 trajectories, same metrics as `intervention_analysis_phase1_20260921`. No generation or external scoring.

Archive: `/storage/yukaichengLab/wangxin/test_eiloeo/opd_analysis/intervention_logits_bf16_greedy_20260921`.
Manifest: `06a3728d5adf10ddd32324d634e331cb8eab4a4990f3909ae3d9a60ec75ca4bb`.
Output: `/storage/yukaichengLab/wangxin/test_eiloeo/opd_analysis/intervention_analysis_greedy_20260921`.

Changes from stochastic analysis: frozen input/output identity and counts, greedy score paths, single-round labeling, absent historical teacher top16 represented as unavailable instead of assuming populated arrays. The original eight-round archive and metrics are untouched. Existing tests and independent FP64 checks retained.

Pipeline: extract.py -> validate.py -> summarize.py -> compare_tokens.py -> continuous_support.py -> validate_mass.py -> teacher_uncertainty.py -> compare_regimes.py.

Important: T=1 softmax measures latent model preference, not greedy action randomness. Historical T=0 argmax versus replay argmax is separately audited. Non-greedy x8 and greedy single-rollout traverse different states, so comparison is descriptive, not a paired causal intervention at identical prefixes. Question-cluster bootstrap accounts for within-question events; there is no greedy seed variability estimate from a single rollout.

Launch history: extraction108237 failed on absent historical top16; partial output retained as `_failed108237`, dependent108238 cancelled. Corrected extraction108240 uses NaN/null for unavailable historical-top16 diagnostics. Final job IDs and completion evidence are recorded alongside the report.

Completed: extraction108240 COMPLETED0:0 in41s; pipeline108242 COMPLETED0:0 in54s. All43879events,4571traces,0missing score matches. Independent FP64 validation39events/546checks plus cumulative-mass checks passed. Report: `reports/intervention_analysis_greedy_20260921/REPORT.md`. All four supplementary analyses and the greedy/non-greedy comparison completed; script hashes checked against completion receipts. Human annotation not performed.
