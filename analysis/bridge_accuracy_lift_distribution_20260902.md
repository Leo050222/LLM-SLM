# Per-bridge accuracy lift distribution (2026-09-02)

## Definition

- One bridge is one `(dataset_index, rollout_index)` pair.
- Each bridge accuracy is estimated from its 8 independent after-bridge
  continuations.
- Its baseline is the 8 direct-Student rollouts for the same prefix.
- Therefore accuracy lift is quantized in steps of 1/8 = 12.5 percentage
  points (pp).
- Total: 13,244 prefixes x 8 bridges = 105,952 bridges.

## Bridge slots 0--7

| Bridge slot | Bridges | Baseline accuracy | After-bridge accuracy | Mean lift | Improved | Unchanged | Decreased |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 13,244 | 39.27% | 44.22% | +4.95 pp | 4,719 | 5,459 | 3,066 |
| 1 | 13,244 | 39.27% | 44.28% | +5.01 pp | 4,727 | 5,413 | 3,104 |
| 2 | 13,244 | 39.27% | 44.26% | +4.99 pp | 4,724 | 5,457 | 3,063 |
| 3 | 13,244 | 39.27% | 44.09% | +4.83 pp | 4,730 | 5,426 | 3,088 |
| 4 | 13,244 | 39.27% | 44.21% | +4.94 pp | 4,655 | 5,491 | 3,098 |
| 5 | 13,244 | 39.27% | 44.04% | +4.77 pp | 4,713 | 5,421 | 3,110 |
| 6 | 13,244 | 39.27% | 44.24% | +4.98 pp | 4,723 | 5,521 | 3,000 |
| 7 | 13,244 | 39.27% | 44.31% | +5.04 pp | 4,733 | 5,420 | 3,091 |

All eight bridge slots have similar mean lift: +4.77 to +5.04 pp.

## Accuracy-lift histogram over all bridges

| Correct-count change (out of 8) | Accuracy change | Number of bridges | Fraction |
|---:|---:|---:|---:|
| -8 | -100.0 pp | 88 | 0.08% |
| -7 | -87.5 pp | 174 | 0.16% |
| -6 | -75.0 pp | 298 | 0.28% |
| -5 | -62.5 pp | 756 | 0.71% |
| -4 | -50.0 pp | 1,434 | 1.35% |
| -3 | -37.5 pp | 3,079 | 2.91% |
| -2 | -25.0 pp | 6,306 | 5.95% |
| -1 | -12.5 pp | 12,485 | 11.78% |
| 0 | 0.0 pp | 43,608 | 41.16% |
| +1 | +12.5 pp | 15,912 | 15.02% |
| +2 | +25.0 pp | 9,066 | 8.56% |
| +3 | +37.5 pp | 5,134 | 4.85% |
| +4 | +50.0 pp | 3,135 | 2.96% |
| +5 | +62.5 pp | 1,744 | 1.65% |
| +6 | +75.0 pp | 1,139 | 1.08% |
| +7 | +87.5 pp | 863 | 0.81% |
| +8 | +100.0 pp | 731 | 0.69% |
| **Total** |  | **105,952** | **100.00%** |

## Three-way summary

| Direction | Bridges | Fraction |
|---|---:|---:|
| Accuracy improved | 37,724 | 35.60% |
| Accuracy unchanged | 43,608 | 41.16% |
| Accuracy decreased | 24,620 | 23.24% |

The bridge-level mean accuracy lift is +4.94 pp. The most common non-zero
change is +12.5 pp (15,912 bridges), followed by -12.5 pp (12,485 bridges).
