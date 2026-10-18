# AIME24 step-0 sampled reverse-KL diagnosis

Run: Slurm `85843`, completed `2026-08-07 16:28:55 +08:00` with exit code `0:0`.

Source:

`/storage/yukaichengLab/wangxin/test_eiloeo/opd_logit_diagnostics/aime24_step0_t1_full_seed20260806`

Generated analysis:

`/storage/yukaichengLab/wangxin/test_eiloeo/opd_logit_diagnostics/aime24_step0_t1_full_seed20260806/sampled_reverse_kl`

## Estimator scope

For a token sampled from the student policy, define

`gap = teacher_chosen_logprob - student_chosen_logprob`.

- `k1 = -gap = log p_S - log p_T`
- `k3 = exp(gap) - 1 - gap`

A single-token `k1` sample may be negative. The expectation over actions sampled
from the student is the conditional reverse KL, `KL(S || T)`. The aggregates here
also average over states visited by the sampled student rollouts. They are sampled
on-policy estimators, not exact full-vocabulary `KL(S || T)` computations.

The sampler used `temperature=1`, `top_p=1`, `top_k=-1`, and `do_sample=true`, so
there is no temperature change or top-p/top-k truncation relative to the raw
student policy. Clipped variants use a symmetric log-ratio limit of 20 only for
numerically bounded reporting; clipping changes the estimator and can introduce
bias.

## Artifact verification

- Slurm state: `COMPLETED`, exit `0:0`, elapsed `00:16:00`.
- 30/30 teacher top-128 NPZ shards and 30/30 student token parquet shards.
- 30 rollouts, 396,297 response tokens, indices exactly 0 through 29.
- Teacher arrays all have width 128 and match each rollout's response length.
- No non-finite chosen log-probabilities or teacher shard values.
- Recomputed chosen-gap maximum absolute error: `2.3841858e-7`.
- `run_config.json` records frozen parameters, no optimizer, no backward, and no
  student parameter requiring gradients.
- Source `tokens.parquet` and `summary.json` retained their original completion
  timestamps after analysis.
- The non-empty Slurm stderr contains progress bars and a deprecation warning;
  it contains no traceback, error, exception, or failure marker.

## Main results

| Weighting | k1 raw mean | k1 std | k3 raw mean | k3 std |
|---|---:|---:|---:|---:|
| Token-weighted | 0.0913372 | 0.503076 | 0.0915361 | 0.966161 |
| Rollout-weighted | 0.0866152 | 0.0244616 | 0.0866453 | 0.0250712 |

The token-weighted standard error is `0.000799` for k1, but it treats correlated
tokens as independent. The more conservative across-rollout standard error is
`0.004466` for the rollout-weighted k1 mean. These are 30 rollouts from one seed,
not 30 independent training seeds.

- 24.7983% of individual k1 samples are negative, while all 30 per-rollout k1
  means are positive.
- k1 raw median is approximately zero (`2.26e-6`), p99 is `2.149`, p99.9 is
  `5.193`, maximum is `22.250`, and minimum is `-5.535`.
- k3 raw median is `3.83e-6`, p99 is `1.713`, p99.9 is `6.264`, and maximum is
  `246.918`.
- k3 exceeds 1 for 2.0099% of tokens, 5 for 0.1655%, and 10 for 0.0447%.
- Clipping at 20 changes the token-weighted mean by only about `5.68e-6` for both
  k1 and k3 on this run.

## Correctness and position

Accuracy is 24/30 (80%). Incorrect rollouts are longer on average and contribute
124,174 tokens across 6 rollouts, versus 272,123 tokens across 24 correct rollouts.

| Group | Token k1 mean | Rollout k1 mean | Token k3 mean | Rollout k3 mean |
|---|---:|---:|---:|---:|
| Incorrect | 0.121561 | 0.124283 | 0.119277 | 0.123437 |
| Correct | 0.077546 | 0.077198 | 0.078877 | 0.077447 |

The incorrect group has a larger reverse-KL estimate under both weighting rules,
but this is descriptive evidence from only six incorrect rollouts and should not
be treated as causal.

Relative-position bins contain nearly equal token counts and all 30 rollouts. The
token-weighted k1 mean is highest in the first 5% (`0.1344`) and lowest in the
last 5% (`0.0419`), with non-monotonic variation in between. Absolute 256-token
bins show the largest stable early means around positions 0–1023; late absolute
bins become noisy as the active-rollout count falls, which is shown explicitly in
the absolute-position figure.

## Extreme tokens

- Largest positive k1: rollout 17, position 4958, token `*`, k1 `22.2499`.
- Most negative k1 / largest k3: rollout 20, position 2662, token ` factors`,
  k1 `-5.53518`, k3 `246.918`.
- Next largest k3 samples are rollout 10 token ` e` (`219.784`) and rollout 29
  token ` chips` (`191.098`).

Full ranked rows, per-rollout tables, position tables, CCDF data, PNG figures, and
an HTML report are in the generated analysis directory.
