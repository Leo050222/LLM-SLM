# Top-1 agreement after one teacher token

Both models predict from identical contexts; raw full-vocabulary argmax at temperature1. All valid k1 intervention positions,8 datasets ×5 thresholds. Terminal positions excluded.

| Dataset | Positions | Top-1 agreement | Trace-equal agreement | Student mean pmax | Teacher mean pmax |
|---|---:|---:|---:|---:|---:|
| ALL | 32360 | 77.69% | 80.60% | 89.00% | 88.09% |
| bbh | 6710 | 73.65% | 76.85% | 89.23% | 88.07% |
| cruxeval_i | 2054 | 63.24% | 67.73% | 92.46% | 95.22% |
| cruxeval_o | 1259 | 59.49% | 70.13% | 89.57% | 93.20% |
| humaneval_plus | 2750 | 86.62% | 89.99% | 94.91% | 96.06% |
| livecodebench | 10013 | 77.17% | 79.90% | 87.45% | 85.02% |
| math500 | 2407 | 84.42% | 89.15% | 85.60% | 83.84% |
| mbpp_plus | 2754 | 85.58% | 87.79% | 95.18% | 97.45% |
| olympiad_bench | 4413 | 82.76% | 84.97% | 84.73% | 81.82% |