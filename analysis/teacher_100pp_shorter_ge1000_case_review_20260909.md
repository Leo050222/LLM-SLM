# +100 pp and Teacher shorter by at least 1,000 tokens: case review

## Scope

- Unit: prefix, with 8 Student and 8 Teacher continuations per prefix.
- Accuracy filter: Student 0/8 and Teacher 8/8, i.e. +100 percentage points.
- Length filter: mean Teacher continuation tokens minus mean Student continuation tokens <= -1,000.
- Clean filter: neither side has any `finish_reason=length` rollout.
- Six prefixes pass all filters. All 96 associated rollouts finish with `stop`.

## Raw summary

| dataset_index | problem_id | Student mean | Teacher mean | Delta | Reduction | Target |
|---:|---|---:|---:|---:|---:|---|
| 974 | deepmath-6167 | 2,569.750 | 640.875 | -1,928.875 | 75.06% | 8pi |
| 1705 | deepmath-10961 | 2,948.125 | 1,541.750 | -1,406.375 | 47.70% | -2 |
| 5004 | deepmath-32360 | 1,640.000 | 594.375 | -1,045.625 | 63.76% | 1 |
| 5484 | deepmath-35477 | 1,610.000 | 584.875 | -1,025.125 | 63.67% | 946 |
| 6613 | deepmath-42830 | 2,004.000 | 547.375 | -1,456.625 | 72.69% | M > 9 |
| 11928 | deepmath-77535 | 2,292.250 | 1,285.625 | -1,006.625 | 43.91% | L sqrt(pi) |

Across the six prefixes, Student averages 2,177.35 continuation tokens and Teacher 865.81, a mean reduction of 1,311.54 tokens (60.24%).

## Case evidence

### 974 / deepmath-6167

The decisive identity is `Delta(phi^2) = 2 phi Delta(phi) + 2 |grad phi|^2`, which gives `Delta(phi)=6`, followed by the divergence theorem and the unit-ball volume. Teacher takes this route and returns `8pi` in all eight rollouts. Student instead changes the requested flux into a flux of `phi grad(phi)`, mis-expands the Laplacian, repeatedly circles around an unknown volume integral, and produces `0` in five of eight runs.

At the fork, Student is forced to continue with token `cdot`, producing a generic `grad dot F` formula; Teacher chooses `phi` with probability 0.999022 in all eight rollouts, directly completing `grad phi dot n`.

### 1705 / deepmath-10961

Student spends roughly three thousand tokens testing assorted values and concludes either no real solution or a domain boundary. Teacher tests the structurally important value `x=-2`, where the two radicals swap pairwise, and returns `-2` in all eight runs. The Teacher traces still do not provide a fully rigorous uniqueness argument, so this case supports better search efficiency more strongly than better proof completeness.

### 5004 / deepmath-32360

The functional `f(x)=x1+...+x10` is nonzero and `M=ker(f)`, hence `C0/M` is isomorphic to `im(f)=R` and has dimension one. Teacher eventually uses codimension-one reasoning. Student repeatedly treats `M` as a nine-dimensional subspace and writes variants of `infinity - 9`, confusing the dimension of the first ten coordinates with the dimension/codimension of a subspace of `C0`.

### 5484 / deepmath-35477

Student performs unguided candidate search and its eight outputs are highly unstable (`183, 154, 1147, 168, 154, 341, 199, 333`). Teacher's first token is `9` with probability 0.918892 in every rollout and it immediately proposes `946`, producing the target eight times. However, representative Teacher traces largely assert that 946 is a known solution or defer to modular computation rather than giving a complete hand verification. This is the clearest case where answer accuracy improves, but proof quality should be audited separately.

### 6613 / deepmath-42830

The correct matrix is `[[1,-3],[-3,M]]`; positive definiteness requires `1>0` and `M-9>0`, hence `M>9`. The prefix stops exactly before the first matrix entry. Student is forced to insert `0`, repeatedly notices contradictions caused by the wrong matrix, and usually concludes no such M. Teacher inserts `1` with probability 0.999973 in all eight rollouts, then finishes using principal minors. This is the cleanest example of a one-token local correction creating both shorter traces and perfect accuracy.

### 11928 / deepmath-77535

Both sides initially continue an algebraically equivalent derivative after the substitution `y=1/x`. Student stays on the nonlinear transformed ODE, attempts an invalid Laplace transform of a product, and finally guesses inconsistent formulas. Teacher recognizes the dead end, switches to energy conservation, obtains `dx/dt=-sqrt(log(L/x))`, and evaluates the hitting-time integral as `L Gamma(1/2)=L sqrt(pi)`. Here the benefit comes from strategy recovery, not merely a locally correct first token.

## Interpretation and caveats

1. The shortening is genuine rather than truncation-driven: all selected rollouts stop normally.
2. The dominant pattern is error-path avoidance. Student uses extra tokens to elaborate an early algebraic mistake, brute-force candidates, or pursue a dead-end transformation; Teacher reaches a decisive identity or invariant sooner.
3. The first fork is unusually influential. Five of six prefixes use the same Teacher first token in all eight Teacher rollouts, while the Student first token is fixed by construction. Therefore, pass-all-8 is conditional on a nearly deterministic early branch and should not be interpreted as eight fully independent discoveries.
4. Selection creates collider/conditioning bias: requiring both +100 pp and a reduction of at least 1,000 tokens deliberately selects examples where Teacher knows a compact successful route and Student is consistently lost. These six cases demonstrate a mechanism but do not estimate its prevalence in the full dataset.
5. Scoring methods differ in the stored score files (`symbolic` for Student and `normalized_exact` for Teacher). The six Teacher answers exactly match the recorded targets and the Student answers do not, so the headline outcome is robust here, but future comparisons should rescore both sides with one common evaluator.

## Suggested checks

- Blindly grade proof correctness/completeness, especially cases 1705 and 5484.
- Rescore all 96 outputs using the same scoring function.
- Repeat the comparison after grouping by first-token pair, to separate model capability from the forced first-token intervention.
