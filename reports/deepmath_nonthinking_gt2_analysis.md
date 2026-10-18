# DeepMath non-thinking analysis with fixed `g_t > 2` threshold

## Definition and data

The analysis fixes the teacher-rejection rule before testing:

```text
g_t = log pi_S(y_t | s_t) - log pi_T(y_t | s_t) > 2
```

This implies `pi_T(y_t | s_t) / pi_S(y_t | s_t) < exp(-2) = 0.1353`.
The data contain 1,000 rollouts and 1,303,870 sampled response tokens, of which
24,249 (1.860%) pass the threshold.  There are 283 correct and 717 wrong
rollouts.  Rollout is the independent statistical unit.

## Position distribution

| Relative position | `g_t > 2` tokens | Share of selected tokens | Rollout-weighted rate | Correct rate | Wrong rate |
|---:|---:|---:|---:|---:|---:|
| 0-10% | 2,074 | 8.55% | 1.64% | 1.27% | 1.78% |
| 10-20% | 3,047 | 12.57% | 2.63% | 1.90% | 2.91% |
| 20-30% | 3,039 | 12.53% | 2.77% | 2.08% | 3.04% |
| 30-40% | 2,861 | 11.80% | 2.72% | 1.97% | 3.01% |
| 40-50% | 2,617 | 10.79% | 2.42% | 1.80% | 2.66% |
| 50-60% | 2,412 | 9.95% | 2.26% | 1.79% | 2.45% |
| 60-70% | 2,325 | 9.59% | 2.17% | 1.54% | 2.42% |
| 70-80% | 2,101 | 8.66% | 1.97% | 1.37% | 2.21% |
| 80-90% | 1,987 | 8.19% | 1.81% | 1.30% | 2.02% |
| 90-100% | 1,786 | 7.37% | 1.53% | 1.00% | 1.75% |

The selected tokens peak at 20-30% of the response, not in the first 10%.
They then decline toward the end.  Paired rollout-level comparisons are:

| Comparison | Left | Right | Difference | Bootstrap 95% CI | Sign-flip p |
|---|---:|---:|---:|---:|---:|
| First 20% vs last 20% | 2.132% | 1.674% | +0.458 pp | [+0.341, +0.577] | 0.000010 |
| First 50% vs last 50% | 2.434% | 1.950% | +0.484 pp | [+0.401, +0.566] | 0.000010 |
| 20-40% vs last 20% | 2.743% | 1.674% | +1.069 pp | [+0.933, +1.207] | 0.000010 |

Among all 24,249 selected tokens, 56.24% lie in the first half and 43.76% in
the second half.  The significance result is based on paired rollout rates,
not on treating tokens as independent observations.

## Correctness association

| Group | Rollouts | Mean rollout `g_t > 2` fraction |
|---|---:|---:|
| Correct | 283 | 1.602% |
| Wrong | 717 | 2.425% |

The wrong-minus-correct difference is `+0.823` percentage points with rollout
bootstrap 95% CI `[+0.659, +0.984]`.  A rollout-label permutation test gives
`p = 0.000020`.  Wrong traces therefore contain 51.4% more `g_t > 2` tokens in
relative terms.  The fraction alone predicts wrong answers with AUROC 0.679,
which is statistically strong but only moderately discriminative.

## Interpretation

With the threshold fixed at 2, both requested effects are significant:

1. Strong teacher rejection is concentrated in the early-middle portion of
   the response and becomes less frequent later.
2. Wrong traces contain substantially more strongly rejected tokens at every
   position decile.

This is an association from one generation seed.  A held-out seed is required
before treating the threshold as a reliable online error detector.
