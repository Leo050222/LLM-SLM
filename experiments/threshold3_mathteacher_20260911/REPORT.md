# Completed teacher direct and absolute-gap threshold3 evaluation

All2814 outputs and scores passed remote audit: completed status, prompt/solution identity, SHA256 score source, score status ok, generation-before-score timestamps. Threshold3 online verified no judge, first chronological absolute gap>3, bridge<=8 and first-token ban. Teacher direct verified original-index seed rule. GPU102590 completed45m03; CPU102591 completed11s. Nonthinking,temp1,top_p1. This experiment used H100; prior baselines mostlyH800.

IMPORTANT: These are historical ABSOLUTE student-minus-teacher logprob gaps. They do not implement the newly clarified positive signed >2 rule. New signed experiments are takeover_ablation_20260911.

| Dataset | Student only | Luna abs>2 teacher8 | NoLuna abs>2 teacher8 | NoLuna abs>3 teacher8 | Teacher14B direct |
|---|---:|---:|---:|---:|---:|
| MATH500 |72.80%|81.40%|84.20%|82.40% (412/500)|87.40% (437/500)|
| OlympiadBench |29.97%|34.42%|46.14%|42.43% (286/674)|46.59%|
| BBH |35.00%|44.81%|75.74%|76.67% (828/1080)|77.31%|
| AIME24 historical |16.67%|10.00%|10.00%|23.33% (7/30)|26.67%|
| AIME25 historical |13.33%|23.33%|20.00%|20.00% (6/30)|26.67%|

No new AIME jobs. AIME rows here were part of the earlier completed run. Scores are single-rollout accuracy, not pass@8/64. Threshold3 vs2 difference: MATH -1.80pp, Olympiad -3.71pp, BBH +0.93pp; single seeded runs do not establish statistically reliable superiority.

Threshold3 mean retained length / mean takeovers / pooled teacher-token share:
MATH500:824.964 /3.924 /3.80165%.
Olympiad:1447.2908 /6.7893 /3.702%.
BBH:270.0546 /8.2167 /23.2192%.
Teacher direct MATH500 mean length911.388.
