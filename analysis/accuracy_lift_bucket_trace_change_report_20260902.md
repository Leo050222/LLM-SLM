# Accuracy-lift bucket vs. whole-trace change (2026-09-02)

## Definition

- Unit: one source bridge, identified by `(dataset_index, rollout_index)`.
- There are 105,952 bridges in total.
- Each bridge has 8 after-bridge continuations; the direct baseline has 8
  rollouts for the same prefix. Accuracy-lift buckets therefore have a 12.5 pp
  interval.
- Trace shortening = direct Student mean whole-trace tokens minus after-bridge
  mean whole-trace tokens. Positive values mean the bridge condition is
  shorter; negative values mean it is longer.
- `Clean` excludes a bridge comparison when either side contains a
  length-finished rollout. This removes 289/105,952 bridge comparisons (0.27%).

## Results by accuracy-lift bucket

| Accuracy lift | Bridges | Length-affected | Raw median shortening | Clean mean shortening | Clean median shortening | Clean shorter fraction |
|---:|---:|---:|---:|---:|---:|---:|
| -100.0 pp | 88 | 6 | -94.44 | -71.22 | -78.75 | 32.93% |
| -87.5 pp | 174 | 1 | -30.13 | -4.06 | -27.25 | 46.24% |
| -75.0 pp | 298 | 0 | -18.75 | -41.10 | -18.75 | 46.31% |
| -62.5 pp | 756 | 0 | -51.75 | -61.75 | -51.75 | 41.80% |
| -50.0 pp | 1,434 | 1 | -56.44 | -67.63 | -56.38 | 41.59% |
| -37.5 pp | 3,079 | 10 | -60.88 | -60.27 | -60.50 | 39.36% |
| -25.0 pp | 6,306 | 5 | -53.63 | -52.17 | -53.38 | 40.18% |
| -12.5 pp | 12,485 | 14 | -41.63 | -30.78 | -41.50 | 42.21% |
| 0 pp | 43,608 | 155 | -32.38 | -22.50 | -32.00 | 43.90% |
| +12.5 pp | 15,912 | 32 | -5.63 | +3.00 | -5.63 | 48.82% |
| +25.0 pp | 9,066 | 15 | +26.00 | +39.13 | +26.13 | 54.58% |
| +37.5 pp | 5,134 | 10 | +42.19 | +68.20 | +42.50 | 57.51% |
| +50.0 pp | 3,135 | 8 | +70.13 | +108.98 | +70.25 | 60.51% |
| +62.5 pp | 1,744 | 2 | +102.56 | +143.68 | +102.63 | 63.83% |
| +75.0 pp | 1,139 | 8 | +77.00 | +139.02 | +77.75 | 61.36% |
| +87.5 pp | 863 | 10 | +69.25 | +141.98 | +73.50 | 63.89% |
| +100.0 pp | 731 | 12 | +21.38 | +118.33 | +24.50 | 54.10% |

## Association strength

At the individual bridge level:

| Data | Pearson r | Spearman rho |
|---|---:|---:|
| All 105,952 bridges | 0.018 | 0.131 |
| Excluding any length finish (105,663 bridges) | 0.134 | 0.132 |

Across the 17 clean bucket means:

| Comparison | Correlation |
|---|---:|
| Accuracy-lift bucket vs. mean trace shortening, Pearson | 0.897 |
| Accuracy-lift bucket vs. mean trace shortening, Spearman | 0.882 |
| Accuracy-lift bucket vs. median trace shortening, Pearson | 0.842 |

The bucket-level association is strong because aggregation averages away much
of the per-bridge sampling noise. It must not be interpreted as an individual
bridge correlation of 0.90; individual bridge association remains weak at
about 0.13.

## Interpretation

The central, well-populated buckets show a clear dose-response pattern. Clean
mean shortening rises from -30.78 tokens at -12.5 pp accuracy lift, to +3.00 at
+12.5 pp, +39.13 at +25 pp, +68.20 at +37.5 pp, +108.98 at +50 pp, and +143.68
at +62.5 pp. The probability that the whole trace is actually shorter rises in
parallel from 42.21% to 63.83%.

The +75/+87.5/+100 pp buckets are smaller and affected by ceiling/boundary
selection: to gain +100 pp, the direct prefix must be 0/8 and the bridge must be
8/8. Their slight non-monotonicity does not reverse the overall trend.

The supported claim is: bridge groups with greater accuracy improvement tend,
on average, to have shorter whole traces. This is an association, not causal
evidence; a bridge that resolves the underlying reasoning bottleneck may cause
both outcomes.
