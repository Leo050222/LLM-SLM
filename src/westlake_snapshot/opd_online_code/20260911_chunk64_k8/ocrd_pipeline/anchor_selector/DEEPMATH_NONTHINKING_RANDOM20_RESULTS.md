# DeepMath native non-thinking random-20 anchor evaluation

Date: 2026-08-23

Sampling: `random.Random(20260823).sample(range(1000), 20)` over all 1,000
Qwen3-1.7B native non-thinking rollouts. Correctness was not used for sample
selection. The sample contains 9 student-correct and 11 student-wrong traces;
the protocol-matched Qwen3-14B is correct on 10 and wrong on 10.

Candidate rule:

\[
\left|\log \pi_S(y_t\mid s_t)-\log \pi_T(y_t\mid s_t)\right|>2.
\]

Candidates were judged chronologically by OpenLux `claude-opus-5`; each trace
stopped at the first accepted `reasoning_branch` or `local_correction`.

## Raw results

Positions are zero-based and relative to the complete student response.

| Order | Index | Student | Teacher | Response tokens | Candidates | Judge calls | Result | Position | Relative | Gap | Confidence |
|---:|---:|:---:|:---:|---:|---:|---:|---|---:|---:|---:|---:|
| 0 | 652 | correct | correct | 1,296 | 45 | 11 | selected | 398 | 30.7% | 3.4394 | 0.60 |
| 1 | 898 | wrong | wrong | 881 | 27 | 3 | selected | 50 | 5.7% | 8.5868 | 0.85 |
| 2 | 510 | correct | correct | 1,610 | 9 | 2 | selected | 792 | 49.2% | 12.3749 | 0.86 |
| 3 | 779 | wrong | wrong | 1,236 | 13 | 7 | selected | 490 | 39.6% | 8.0001 | 0.83 |
| 4 | 141 | correct | correct | 620 | 11 | 7 | selected | 304 | 49.0% | 3.1611 | 0.58 |
| 5 | 788 | correct | correct | 1,141 | 9 | 9 | no anchor | — | — | — | — |
| 6 | 300 | correct | wrong | 681 | 7 | 7 | no anchor | — | — | — | — |
| 7 | 59 | wrong | correct | 2,923 | 34 | 9 | selected | 633 | 21.7% | 5.5062 | 0.76 |
| 8 | 985 | wrong | wrong | 1,778 | 51 | 9 | selected | 294 | 16.5% | 2.3851 | 0.58 |
| 9 | 310 | wrong | correct | 2,041 | 18 | 6 | selected | 938 | 46.0% | 3.6551 | 0.66 |
| 10 | 637 | wrong | wrong | 1,223 | 36 | 5 | selected | 356 | 29.1% | 2.9833 | 0.72 |
| 11 | 635 | wrong | wrong | 1,602 | 59 | 17 | selected | 359 | 22.4% | 4.8945 | 0.88 |
| 12 | 536 | correct | correct | 1,318 | 70 | 8 | selected | 155 | 11.8% | 4.2459 | 0.70 |
| 13 | 755 | correct | correct | 938 | 18 | 10 | selected | 315 | 33.6% | 3.4135 | 0.55 |
| 14 | 278 | wrong | wrong | 2,013 | 55 | 12 | selected | 359 | 17.8% | 3.3199 | 0.62 |
| 15 | 156 | correct | correct | 1,547 | 23 | 2 | selected | 62 | 4.0% | 3.5313 | 0.85 |
| 16 | 396 | correct | correct | 1,017 | 22 | 8 | selected | 148 | 14.6% | 7.0565 | 0.72 |
| 17 | 899 | wrong | wrong | 1,628 | 45 | 5 | selected | 230 | 14.1% | 14.7871 | 0.55 |
| 18 | 692 | wrong | wrong | 916 | 25 | 5 | selected | 281 | 30.7% | 3.1593 | 0.85 |
| 19 | 849 | wrong | wrong | 1,736 | 20 | 20 | selected | 1,656 | 95.4% | 2.0942 | 0.60 |

## Aggregate results

| Metric | Value |
|---|---:|
| Selected anchors | 18/20 (90.0%) |
| No anchor after exhaustive judging | 2/20 (10.0%) |
| Student-wrong traces selected | 11/11 (100%) |
| Student-correct traces selected | 7/9 (77.8%) |
| `reasoning_branch` selections | 16 |
| `local_correction` selections | 2 |
| Raw candidates | 597 |
| Valid recorded judge calls | 162 |
| Candidates not requested because of early stopping | 435 (72.9%) |
| Recorded API tokens | 333,085 |
| Mean / median judge calls per trace | 8.1 / 7.5 |
| Mean / median selected relative position | 29.5% / 25.8% |
| Mean selected gap | 5.3663 |
| Mean judge confidence | 0.7089 |

Three additional malformed judge replies occurred before the client repair and
are not represented in the recorded API-token total. Their candidates were
retried without duplicating any saved valid judgment.

Candidate density is nearly identical after response-length normalization:
21.05 candidates per 1,000 tokens on correct traces and 21.31 on wrong traces.
Mean selected gap is also similar (5.32 correct versus 5.40 wrong), as is mean
selected position (27.5% versus 30.8%).

## Interpretation

The current selector is effective at finding a plausible point where teacher
and student can follow different reasoning, but it is not an error detector.
Seven of nine correct traces also contain an accepted anchor. Several are
legitimate alternative strategies that still reach the correct answer, such
as simplifying a denominator versus a numerator (trace 156), substitution
versus a parameterized-integral method (trace 396), or change of variables
versus a squeeze bound (trace 755). Correct traces 510 and 141 contain a local
mistake that the student later repairs, so a local correction can exist without
changing final correctness.

At least one accepted case remains questionable: trace 692 is largely a choice
between mathematical notation and prose, despite being labeled a reasoning
branch. This demonstrates residual judge false positives after the `|g_t|`
filter.

The observed 100% versus 77.8% selection rates for wrong and correct traces are
suggestive but not conclusive at `n=20`; the two no-anchor cases both happen to
be correct, whose one-sided allocation probability under equal selection is
`36/190 = 0.1895`.

For the intended causal pipeline, stage 3 must evaluate both sides:

- wrong trace: whether teacher bridge plus student continuation changes the
  final answer from wrong to correct;
- correct trace: whether the intervention preserves correctness rather than
  damaging a valid student route.

If the desired anchor definition is specifically "likely error-causing
divergence," the judge prompt needs an additional causal-error criterion. The
current prompt intentionally accepts any substantive branch or local
correction, including valid alternatives on correct traces.

## Operational evidence

- Initial H800 array: `92140`; tasks 0--16 completed. Tasks 17--19 failed before
  model load on `gnho009` because its NVIDIA driver was too old.
- Recovery H800 array: `92161_17..19`, excluding `gnho009` and `gnho018`; all
  completed `0:0`.
- Batch judging exposed malformed Opus JSON on three calls. The client now
  retries JSON parse/schema failures using the existing retry budget; all five
  unit tests pass.
- Local artifact verification checked 20 unique traces, ordered unique
  judgments, trace hashes, accepted positions, and exhaustive no-anchor cases.

Remote artifact root:

```text
/storage/yukaichengLab/wangxin/test_eiloeo/ocrd_anchor_nonthinking_random20_seed20260823
```

Local artifact root:

```text
/Users/leolee/Documents/OPD/tmp/anchor_nonthinking_random20_seed20260823/runs
```
