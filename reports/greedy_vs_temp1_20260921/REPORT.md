# Greedy vs temperature=1 baselines

All scores complete. Identical frozen prompt hashes, models and generation token limits. Greedy has one completion/question; temperature=1 has eight seeds. Metric: single-completion accuracy (temperature=1 mean over seeds), NOT pass@8. ± denotes sample SD across eight temperature=1 seeds, not a confidence interval. Δ is greedy minus temperature=1 in percentage points.

| Dataset | Role | Greedy correct/N | Greedy % | T=1 mean ± SD % | Δ pp | Relative change % |
|---|---|---:|---:|---:|---:|---:|
| cruxeval_i | student | 159/800 | 19.88 | 19.36 ± 0.71 | +0.52 | +2.66 |
| cruxeval_i | teacher | 393/800 | 49.12 | 49.30 ± 1.02 | -0.17 | -0.35 |
| cruxeval_o | student | 186/800 | 23.25 | 21.84 ± 0.45 | +1.41 | +6.44 |
| cruxeval_o | teacher | 370/800 | 46.25 | 45.81 ± 0.54 | +0.44 | +0.95 |
| humaneval_plus | student | 106/164 | 64.63 | 62.96 ± 1.95 | +1.68 | +2.66 |
| humaneval_plus | teacher | 139/164 | 84.76 | 84.68 ± 1.86 | +0.08 | +0.09 |
| mbpp_plus | student | 198/378 | 52.38 | 51.88 ± 1.21 | +0.50 | +0.96 |
| mbpp_plus | teacher | 275/378 | 72.75 | 73.21 ± 0.80 | -0.46 | -0.63 |
| livecodebench | student | 34/175 | 19.43 | 17.79 ± 1.31 | +1.64 | +9.24 |
| livecodebench | teacher | 50/175 | 28.57 | 29.29 ± 1.80 | -0.71 | -2.44 |
| math500 | student | 360/500 | 72.00 | 70.55 ± 1.22 | +1.45 | +2.06 |
| math500 | teacher | 418/500 | 83.60 | 83.80 ± 0.74 | -0.20 | -0.24 |
| olympiad_bench | student | 283/674 | 41.99 | 40.75 ± 0.96 | +1.24 | +3.05 |
| olympiad_bench | teacher | 401/674 | 59.50 | 58.72 ± 1.17 | +0.78 | +1.33 |
| bbh | student | 474/1080 | 43.89 | 43.03 ± 0.67 | +0.86 | +1.99 |
| bbh | teacher | 866/1080 | 80.19 | 79.88 ± 0.60 | +0.30 | +0.38 |

Overall is weighted by question count, not equal-weight dataset averaging.
- student: greedy 39.38%; T=1 38.33 ± 0.21%; Δ +1.04 pp; relative +2.73%.
- teacher: greedy 63.71%; T=1 63.56 ± 0.36%; Δ +0.15 pp; relative +0.23%.

1. Student improves on all eight datasets; teacher changes are mixed and small.
2. Same 900-second active-wall-time deadline and timeout-count-wrong grading policy, but deployment differs: greedy began on two GPUs then moved to eight; T=1 used four then eight. These operational results do not isolate temperature causally.
3. Follow-up if needed: rerun both temperatures with matched hardware/concurrency and paired per-question analysis; this report does not assert statistical significance.

Score rows validated against summaries; timeout counts/rates:
- greedy student: 125/4571 = 2.73% timed out.
- temp1 student: 250/36568 = 0.68% timed out.
- greedy teacher: 121/4571 = 2.65% timed out.
- temp1 teacher: 36/36568 = 0.10% timed out.
