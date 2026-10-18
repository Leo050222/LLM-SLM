# DeepMath wrong-trace anchor smoke test

Date: 2026-08-22

Judge provider/model: OpenLux `claude-opus-5`

Candidate rule:

\[
\left|\log \pi_S(y_t\mid s_t)-\log \pi_T(y_t\mid s_t)\right|>2.
\]

| Dataset index | Error | Response tokens | Raw candidates | Judged before selection | Selected position | Gap | Judge confidence |
|---:|---|---:|---:|---:|---:|---:|---:|
| 338 | `No` vs `Yes` | 6,676 | 294 | 6 | 68 | 7.4254 | 0.60 |
| 216 | `No` vs `510511` | 10,786 | 254 | 9 | 146 | 9.2488 | 0.62 |
| 485 | `k=1 or k=n-1` vs `gcd(n,k)=1` | 17,502 | 540 | 29 | 346 | 2.7180 | 0.62 |

## Selected anchors

### Trace 338

The prefix ends with the student's working definition of a subring:
`closed under addition, multiplication, and contains the`. The student token is
` identity`, while the teacher assigns probability 0.9792 to ` additive` and
0.00042 to ` identity`. The judge treats this as a substantive choice between
unital and non-unital subring conventions.

### Trace 216

The prefix ends with `So, for example, E_1 is 2`. The student token is `×`,
while the teacher assigns probability 0.9931 to ` +` and 0.000095 to `×`.
This is the start of the off-by-one Euclid-number indexing error: the student
continues the prime product instead of writing `E_1 = 2 + 1`.

### Trace 485

The prefix ends with `For k=1: S_{2,1} is {v1`. The student token is ` +`,
while the teacher assigns probability 0.9112 to `,` and 0.0660 to ` +`.
The student therefore initially reads the k=1 member as a multi-vector sum,
whereas the teacher enumerates the original basis `{v1, v2}`.

## Interpretation

The raw gap threshold has useful recall but low precision: it produced hundreds
of candidates per trace, and most early candidates were surface/tokenization
differences. The model judge rejected 5, 8, and 28 candidates respectively
before selecting an anchor.

The current stopping rule finds the first plausible logical divergence. That is
not guaranteed to be the divergence that causes the final wrong answer. Trace
485 is the clearest example: the selected k=1 interpretation is a real local
reasoning branch, but a later circulant-matrix/generalization error may be more
causally related to the final `k=1 or k=n-1` answer. Bridge evaluation should
therefore measure whether intervening at the selected anchor actually changes
the downstream answer, rather than treating judge acceptance alone as proof of
causal relevance.
