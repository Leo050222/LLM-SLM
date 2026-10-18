# Cross-round KV reuse

The current local online worker enables vLLM automatic prefix caching by default.
`--disable-prefix-caching` creates a separate full-prefill reference worker. Use
new queue directories: model identity now covers both worker and sidecar code,
cache configuration and raw-logprob mode. Historical experiment snapshots are
unchanged; their existing results were generated without this optimization.

For prompt scoring, `skip_reading_prefix_cache=False` must be explicit. In the
installed vLLM 0.23.0, merely enabling prefix caching does not override the
prompt-logprob request's default skip behavior.

The model-local CPU sidecar retains raw chosen/top-16 logprobs under a hash of
all prefix tokens plus the target token. Student drafts can be scored without
another GPU forward. Teacher continuation scoring computes the new suffix and
uses sidecar records for positions omitted on a KV hit. Rollback branches share
only their unchanged prefix. Token bans affect selection, while logged scores
remain the unmasked raw distribution.

KV blocks do not include boundary logits. If the first new target is predicted
by an already cached hidden state, a one-token constrained decode recovers its
raw score using cached preceding KV. vLLM recomputes the final token/block as
needed; the repair is explicit in `boundary_repairs` and its separate duration.
Unknown prompt-logprob layouts or missing uncached scores fail loudly rather
than silently rerunning the entire prefix.

Metrics distinguish `computed_prompt_tokens`, `num_cached_tokens`, logprob-only
sidecar hits and boundary repair work. Total prefill work must include the
nested repair counters. Whole-batch time is shared across batch members and
must not be summed across requests in the same batch.

Automatic prefix caching remains subject to GPU cache eviction and process
lifetime. A cold/evicted prefix must be recomputed; this is not a pinned KV
session guarantee. Cache counters must be monitored when increasing active
trace concurrency. The CPU logprob sidecar currently lives until worker exit.

BF16 decode and full-prefill can differ numerically. The original strict 0.05
logprob check failed (observed maximum 0.249775); a functional cache test is not
proof of numerical equivalence or unchanged accuracy near the intervention
threshold. The v2 test records that failure explicitly and measures a no-cache
decode/prefill control. A separate matched batch-one microbenchmark measures
warm teacher scoring only, not total rollout speed.

Local validation: 21 online tests passed, including cached-slot alignment,
branch isolation, missing boundary detection/repair, repeat takeovers and EOS.
See VERIFY.md for the independent live invocation and remote evidence location.

## Live result (job 102555, four H100 workers)

The independent documented invocation completed successfully after fixing a
validation-script tuple-key typo. It exercised 22 GPU prefix-cache hits,
student sidecar reuse and block-boundary repair. Maximum cached/reference
chosen-logprob difference was 0.3407936096, so the original 0.05 tolerance
remains FAILED. No-cache decode-vs-prefill control maxima were 0.2166461945
(student) and 0.0996198654 (teacher); mean absolute differences were 0.0076672
and 0.0098884 respectively. These controls demonstrate numerical drift exists
without caching; they do not explain away every difference or prove unchanged
intervention decisions.

The test's aggregate cached/reference prefill counters cover different mixes
of generation and scoring; do not divide them to claim a speedup. The separate
benchmark uses matched teacher score requests, eight incremental 64-token
chunks per context length, and counts boundary repair work. Its time ratio is
engine.generate wall-clock time (including host work), not CUDA-event kernel
time or end-to-end rollout speed. Synthetic prompts, warm cache, batch one.

Measured warm teacher score-call speed ratios (full prefill / KV reuse):

| Initial prefix | Chunk | Ratio | Computed prompt tokens, cached / reference (8 rounds) |
| --- | --- | --- | --- |
| 1,024 | 64 | 9.05x | 536 / 10,520 |
| 4,096 | 64 | 27.84x | 536 / 35,096 |
| 8,192 | 64 | 47.73x | 536 / 67,864 |

The cached path averaged 67 computed prompt tokens per 64-token extension,
including the partial-block tail in this benchmark. Warmup is excluded.
Remote evidence is validation.json and benchmark.json under
`opd_eval_deepmath/kv_reuse_validation_20260911_v2`. Validated worker snapshot:
`opd_online_code/20260911_kv_reuse_v2`. STOP sent to both validation queues after
benchmark completion; no new 1,000-question accuracy experiment was launched.

Slurm confirmed both validation jobs COMPLETED 0:0: 102548 (6m11s) and
102555 (5m10s). The validation allocations have been released.
