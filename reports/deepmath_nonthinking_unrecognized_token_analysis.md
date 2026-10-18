# DeepMath non-thinking teacher-rejected token analysis

## Setup

- Student: frozen Qwen3-1.7B native non-thinking rollout.
- Teacher: frozen Qwen3-14B scored on the actual student-sampled token.
- Metric: `g_t = log pi_S(y_t | s_t) - log pi_T(y_t | s_t)`.
- A teacher-rejected token is operationalized as a positive gap above one of
  three thresholds: page strong-S (`g_t > 0.585700`), global p95 absolute-gap
  scale (`g_t > 1.171400`), or strict (`g_t > 2`).
- Coverage: 1,000 rollouts, 1,303,870 tokens; 283 correct and 717 wrong.
- Primary statistical unit: rollout.  Position tests use paired sign-flip
  permutation tests; correctness tests permute rollout labels.  Confidence
  intervals are rollout bootstraps.  Reported p-values are Holm-adjusted over
  the three thresholds.

## Relative-position distribution

Rollout-weighted teacher-rejected-token rate by response decile:

| Relative position | `g_t > 0.5857` | `g_t > 1.1714` | `g_t > 2` |
|---:|---:|---:|---:|
| 0-10% | 7.06% | 3.59% | 1.64% |
| 10-20% | 10.12% | 5.23% | 2.63% |
| 20-30% | 10.62% | 5.57% | 2.77% |
| 30-40% | 10.70% | 5.52% | 2.72% |
| 40-50% | 10.32% | 5.17% | 2.42% |
| 50-60% | 10.16% | 4.99% | 2.26% |
| 60-70% | 10.04% | 4.89% | 2.17% |
| 70-80% | 9.83% | 4.65% | 1.97% |
| 80-90% | 9.82% | 4.42% | 1.81% |
| 90-100% | 8.14% | 3.71% | 1.53% |

| Threshold | First 50% | Last 50% | Difference | Bootstrap 95% CI | Holm p |
|---|---:|---:|---:|---:|---:|
| `g_t > 0.5857` | 9.77% | 9.60% | +0.17 pp | [-0.05, +0.40] | 0.148 |
| `g_t > 1.1714` | 5.02% | 4.53% | +0.48 pp | [+0.35, +0.62] | <0.00003 |
| `g_t > 2` | 2.43% | 1.95% | +0.48 pp | [+0.40, +0.57] | <0.00003 |

The distribution is hump-shaped rather than monotonically front-loaded.  All
three definitions peak around 20-40% of the response and then decline.  For
the two stricter definitions, the first half has a significant excess over the
second half.  At the broad page-color threshold, the half-vs-half difference
is not significant because moderately rejected tokens remain common through
the middle and late response.

## Association with correctness

| Threshold | Correct traces | Wrong traces | Wrong - correct | Bootstrap 95% CI | Holm p | AUROC for wrong |
|---|---:|---:|---:|---:|---:|---:|
| `g_t > 0.5857` | 7.78% | 10.43% | +2.66 pp | [+2.15, +3.17] | <0.00006 | 0.685 |
| `g_t > 1.1714` | 3.65% | 5.22% | +1.57 pp | [+1.28, +1.87] | <0.00006 | 0.685 |
| `g_t > 2` | 1.60% | 2.42% | +0.82 pp | [+0.66, +0.98] | <0.00006 | 0.679 |

Wrong traces have a higher teacher-rejected-token fraction at every relative
position decile and under every tested threshold.  The association is highly
significant but only moderately discriminative on its own (AUROC about 0.68),
so it is a useful risk signal rather than a standalone error detector.

## Limitations and next test

- These are observational associations from one generation seed, not evidence
  that rejected tokens cause the final error.
- The analysis uses the compact int16 visualization payload.  Its quantization
  error is negligible at these thresholds, but exact float32 scores should be
  used for a paper table.
- The next useful test is a preregistered rollout-level predictor using only
  the first 40% of tokens, evaluated on a new seed or held-out split, and
  compared with response length and problem difficulty baselines.
