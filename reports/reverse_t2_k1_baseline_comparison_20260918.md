# Reverse intervention vs single-model baselines

Accuracy (%), generation timeout counted incorrect. Historical single-rollout baselines; not an x8 aggregate or paired-seed significance test. Prompt token equality and score source hashes verified; identical GPT-4o-mini judge prompt and code score_one.py. Reverse seed 20260912; historical code baseline seed 20260911.

| Dataset | Student only | Teacher only | Reverse | Δ teacher (pp) |
|---|---:|---:|---:|---:|
| cruxeval_o | 21.88 | 46.75 | 23.50 | -23.25 |
| cruxeval_i | 19.75 | 50.75 | 23.25 | -27.50 |
| humaneval_plus | 62.80 | 87.20 | 63.41 | -23.78 |
| mbpp_plus | 51.85 | 72.49 | 57.41 | -15.08 |
| livecodebench | 17.71 | 30.86 | 17.14 | -13.71 |
| olympiad_bench | 41.69 | 55.04 | 47.18 | -7.86 |
| bbh | 43.06 | 79.81 | 42.50 | -37.31 |
| math500 | 71.60 | 84.60 | 76.40 | -8.20 |
| Total (sample-weighted) | 38.66 | 63.60 | 41.22 | -22.38 |

Correct counts: student 1767, teacher 2907, reverse 1884; reverse-student +2.56 pp.
Reverse timeout count 244; scoring partial answers without timeout penalty recovers only 10 correct answers.
