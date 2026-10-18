| Dataset | Student greedy | Clean correction greedy | Teacher greedy | Delta pp |
|---|---:|---:|---:|---:|
| bbh | 43.89 | 34.07 | 80.19 | -9.81 |
| cruxeval_i | 19.88 | 7.25 | 49.12 | -12.62 |
| cruxeval_o | 23.25 | 21.75 | 46.25 | -1.50 |
| humaneval_plus | 64.63 | 64.63 | 84.76 | +0.00 |
| livecodebench | 19.43 | 14.29 | 28.57 | -5.14 |
| math500 | 72.00 | 76.20 | 83.60 | +4.20 |
| mbpp_plus | 52.38 | 48.68 | 72.75 | -3.70 |
| olympiad_bench | 41.99 | 49.41 | 59.50 | +7.42 |
| Overall | 39.38 | 35.64 | 63.71 | -3.74 |

Same 4571 questions; all greedy single rollouts. Baseline trace deadline900s vs correction2000s; deployment/concurrency differ. Timeouts count wrong.
