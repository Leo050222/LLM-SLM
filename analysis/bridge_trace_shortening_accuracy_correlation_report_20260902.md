# Bridge trace shortening vs. accuracy lift (2026-09-02)

## Question and units of analysis

This analysis asks whether the change from `Student from prefix` to `Student
after a 256-token teacher bridge` has an association between:

- whole-trace reduction = direct Student whole-trace length minus after-bridge
  whole-trace length (positive means shorter after the bridge); and
- accuracy lift = after-bridge accuracy minus direct Student accuracy for the
  same prefix.

Two aggregation levels are reported:

1. Bridge level: 105,952 source bridges. Each bridge's after value averages its
   8 replicate continuations and is compared with the 8 direct Student rollouts
   for the corresponding prefix.
2. Prefix level: 13,244 prefixes. Each after value averages 8 bridges x 8
   replicate continuations (64 samples) and is compared with the 8 direct
   Student rollouts.

## Correlation results

| Level | N | Pearson r | Spearman rho | Pearson r, excluding any length finish | Spearman rho, excluding any length finish |
|---|---:|---:|---:|---:|---:|
| Bridge | 105,952 | 0.018 | 0.131 | 0.134 | 0.132 |
| Prefix | 13,244 | 0.014 | 0.118 | 0.132 | 0.123 |

The raw Pearson correlation is close to zero because a small number of
near-40.9k length-truncated traces create extreme negative shortening values.
Rank correlation is robust to these values and shows a weak positive monotonic
association. On the 13,070 prefixes without any length finish (98.69% of all
prefixes), Pearson and Spearman agree at about 0.12--0.13.

Controlling linearly for direct-Student baseline accuracy and baseline length:

- prefix-level partial Pearson is 0.035 on all data;
- it is 0.200 after excluding prefixes containing a length finish.

Thus the association is not only the trivial consequence of sharing the same
baseline, although its effect size remains modest and tail-sensitive.

## Shorter versus not-shorter prefixes

| Prefix group | Prefixes | Fraction | Mean trace reduction | Mean accuracy lift | Probability accuracy improves | Mean final accuracy |
|---|---:|---:|---:|---:|---:|---:|
| After-bridge trace is shorter | 5,905 | 44.59% | +192.93 tokens | +6.94 pp | 55.61% | 45.86% |
| After-bridge trace is not shorter | 7,339 | 55.41% | -248.44 tokens | +3.33 pp | 47.15% | 42.87% |

Shorter prefixes have:

- 3.61 percentage points more accuracy lift on average; and
- an 8.47 percentage-point higher probability of positive accuracy lift.

Both groups still improve on average, so shortening is not necessary for an
accuracy gain. Conversely, only 55.61% of shortened prefixes improve, so
shortening is not sufficient either.

Joint proportions over all prefixes:

| Outcome | Fraction |
|---|---:|
| Shorter and accuracy improves | 24.80% |
| Shorter and accuracy does not improve | 19.79% |
| Not shorter and accuracy improves | 26.13% |
| Not shorter and accuracy does not improve | 29.29% |

## Dose-response by shortening quintile

| Trace-reduction quintile | Mean reduction | Mean accuracy lift | Probability accuracy improves |
|---|---:|---:|---:|
| Most lengthened 20% | -552.82 tokens | +2.80 pp | 45.98% |
| Q2 | -110.92 | +3.32 pp | 47.36% |
| Q3 | -22.24 | +3.77 pp | 48.06% |
| Q4 | +75.87 | +5.80 pp | 53.30% |
| Most shortened 20% | +351.83 | +9.01 pp | 59.91% |

Accuracy lift rises monotonically across the five shortening quintiles. The
most-shortened quintile gains 6.21 pp more accuracy than the most-lengthened
quintile. This monotonic pattern is the clearest evidence for a positive
association.

Final accuracy itself is not monotonic across the bins because prefix
difficulty and baseline accuracy differ. The claim supported here is about
*accuracy lift relative to each prefix's own baseline*, not that shorter traces
universally have the highest absolute accuracy.

## Interpretation

There is a reproducible but weak positive association: bridges that allow the
student to finish with a shorter whole trace tend to produce a larger accuracy
gain. It is not a strong per-prefix predictor, and the observational comparison
does not establish that shortening causes correctness. A plausible common
cause is that a useful bridge resolves the key reasoning bottleneck, which both
reduces subsequent meandering and improves the answer.

## Suggested next experiments

1. Add bridge length interventions (for example 64/128/256/512 tokens) on the
   same prefixes and seeds. This can distinguish causal bridge dose effects
   from mere outcome correlation.
2. Analyze the 174 prefixes containing length-finished runs separately; they
   dominate raw Pearson statistics and likely represent looping/failure modes.
3. Fit a prefix-level model predicting accuracy lift from shortening while
   controlling for baseline correctness, baseline length, difficulty, topic,
   and anchor position, with held-out-prefix evaluation.

Reproduction script:
`analysis/bridge_trace_shortening_accuracy_correlation_20260902.py`.
