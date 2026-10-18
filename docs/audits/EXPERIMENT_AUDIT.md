# AIME24 answer-extraction audit

**Date:** 2026-08-07  
**Auditor:** fresh GPT-5.6-Sol reviewer, same-family provisional  
**Scope:** Slurm job 85843, 30 AIME24 rollouts

## Answer-accounting verdict: PASS

- Stored extraction and labels reproduced exactly: 30/30.
- Mathematical final-answer accuracy: 24/30 (80%).
- Format-induced false negatives: 0.
- Extraction/scoring false positives: 0.
- Missing or ambiguous post-thinking boxed answers: 0.
- Failures caused by `max_tokens` or `max_model_len`: 0/6.

The six genuine wrong answers are:

| Dataset index | Problem ID | Extracted | Target |
|---:|---:|---:|---:|
| 2 | 62 | 139 | 371 |
| 3 | 63 | 16 | 385 |
| 13 | 73 | 287 | 197 |
| 21 | 81 | 108 | 315 |
| 28 | 88 | 7 | 127 |
| 29 | 89 | 2 | 902 |

All 30 completions have `finish_reason="stop"`, `stop_reason=null`, end with
`<|im_end|>`, and contain complete post-`</think>` boxed answers. The longest
completion is 28,480 tokens versus `max_tokens=38,912`; the largest combined
prompt and completion is 28,908 versus `max_model_len=40,960`.

## Experiment-audit checks

- **A — PASS:** Ground truth comes from the dataset `solution` field and all 30
  stored problems and solutions match the source parquet.
- **B — PASS:** Accuracy is exact normalized equality; no prediction-derived
  normalization is used.
- **C — PASS:** Artifact counts and reported totals are internally consistent.
- **D — PASS:** The scorer is called in the generation path and its labels feed
  the summary.
- **E — WARN:** This is one rollout per problem, one seed, and one model pair.
- **F — real_gt:** Dataset-provided AIME competition answers.

## Caveats

The scorer has general edge-case risks: an unboxed correct answer is rejected,
equivalent non-integer forms are not symbolically compared, and the `\,`
normalization order is imperfect. None of these affected the 30 audited
responses.
