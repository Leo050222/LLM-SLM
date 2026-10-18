# DeepMath native non-thinking wrong-trace anchor smoke

Date: 2026-08-23

Student rollout model: Qwen3-1.7B with native thinking disabled

Scoring teacher: Qwen3-14B

Judge provider/model: OpenLux `claude-opus-5`

Candidate rule:

\[
\left|\log \pi_S(y_t\mid s_t)-\log \pi_T(y_t\mid s_t)\right|>2.
\]

The three smoke cases were selected from the 202 examples for which the
non-thinking Qwen3-1.7B rollout was wrong and the protocol-matched
non-thinking Qwen3-14B rollout was correct.

| Dataset index | Error | Response tokens | Raw candidates | Judge calls | Selected position (0-based) | Relative position | Gap | Confidence |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 573 | `32` vs `2^{32}` | 330 | 11 | 10 | 233 | 70.6% | 11.7985 | 0.72 |
| 471 | `90` vs `1` | 472 | 13 | 9 | 274 | 58.1% | 5.7360 | 0.56 |
| 251 | `1` vs `1/2` | 519 | 12 | 12 | 444 | 85.5% | 2.6744 | 0.68 |

All three H800 array tasks in Slurm job `92018` completed with exit code
`0:0`. The judge used 31 calls and 61,021 total API tokens before the three
selections.

## Selected anchors

### Trace 573: propositional-logic truth-table count

The prefix ends while stating the number of distinct truth tables over `n`
variables:

```text
For a set of variables with n variables, the number of distinct logical
statements (i.e., distinct truth tables) is 2
```

The student token is `^n` (`p_S=0.817400`), committing to `2^n`. The teacher
assigns that token probability `0.00000614` and assigns probability `0.999897`
to `^{`, which can open the correct nested exponent `2^{2^n}`. This is the
strongest of the three selected anchors and directly matches the final error.

### Trace 471: cotangent product identity

The prefix ends at the numerator of an asserted closed-form identity:

```text
\prod_{k=1}^{n-1} \cot \frac{k\pi}{2n} = \frac
```

The student token is `{n` (`p_S=0.762753`), committing to an `n`-dependent
numerator. The teacher assigns it probability `0.002462`; its top tokens are
`{` (`0.876540`) and `{\\` (`0.118627`). The judge accepted this as the start
of the incorrect identity, but confidence is only `0.56`: the teacher could
still emit `n` after the single-token `{`. This case exposes a tokenization
granularity false-positive risk and should be validated with bridge lookahead.

### Trace 251: heat equation at a discontinuity

The prefix has asserted that `u(1,t)` approaches the point value
`u(1,0)=1`, and ends at:

```text
### Step 4:
```

The student token is ` Conclusion` (`p_S=0.415281`), closing the argument.
The teacher assigns it probability `0.028634`; its top alternatives include
` Use` (`0.211575`), ` Formal` (`0.088197`), and ` Rig` (`0.041662`), which
can open a heat-kernel verification and recover the boundary value `1/2`.
This is a plausible but late intervention point.

## Interpretation

Disabling thinking shortened the sampled traces and reduced the raw candidate
sets from hundreds in the earlier long-thinking smoke to 11--13 here. However,
the first accepted anchors moved to 58--86% of the response, and 31 judge calls
were still required for three traces because almost all earlier candidates were
surface differences.

The method found the causal formula error very cleanly in trace 573. Trace 251
is also semantically plausible, though late. Trace 471 demonstrates why a
next-token top-16 comparison is not sufficient by itself: different tokenizer
segmentations can resemble a logical branch. Stage-2 bridge generation and
stage-3 student continuation should be treated as the causal test.

## Artifacts

Remote root:

```text
/storage/yukaichengLab/wangxin/test_eiloeo/ocrd_anchor_nonthinking_smoke_20260823
```

Local judge artifacts:

```text
/Users/leolee/Documents/OPD/tmp/anchor_nonthinking_smoke_20260823/runs
```
