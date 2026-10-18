# Chris Club mini-judge consistency with cached Luna

Date: 2026-08-24

## Protocol

- Reference: the 200 cached Luna candidate judgments in
  `dev_natural_200_seed20260826`.
- Natural prevalence: 25/200 Luna-positive candidates.
- Input: exact full decoded prefix, problem, sampled student token, gap, and
  teacher top-16 distribution.  The audit export did not retain the student's
  numeric token id, so the replay prompt uses `-1`; log probabilities are
  reconstructed from the retained probabilities.  The semantic evidence and
  all decoded text are unchanged.
- Prompt/rubric: current anchor-selector system prompt; temperature 0,
  `max_tokens=1200`, no explicit reasoning effort.
- Serving: Chris Club OpenAI-compatible endpoint, 32 concurrent requests per
  model; the two models ran concurrently (64 requests total).
- Stop rule: do not spend on balanced held-out when natural-dev agreement is
  plainly below the replacement range.

## Raw results

| Judge | Selected | TP | TN | FP | FN | Accuracy | Precision | Recall | F1 | Cohen kappa |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Luna reference | 25 | — | — | — | — | — | — | — | — | — |
| Qwen3.5-27B careful non-thinking (prior) | 32 | 18 | 161 | 14 | 7 | 0.895 | 0.5625 | 0.7200 | 0.6316 | 0.5714 |
| GPT-5.4-mini | 146 | 22 | 51 | 124 | 3 | 0.365 | 0.1507 | 0.8800 | 0.2573 | 0.0558 |
| GPT-4.1-mini | 107 | 21 | 89 | 86 | 4 | 0.550 | 0.1963 | 0.8400 | 0.3182 | 0.1449 |

GPT-5.4-mini and GPT-4.1-mini agreed with each other on 127/200 candidates
(63.5%, kappa 0.2457).  Both selected 90 candidates; GPT-5.4-mini alone
selected 56 and GPT-4.1-mini alone selected 17.

## Usage and throughput

| Judge | Failed | Prompt tokens | Completion tokens | Reasoning tokens | Total tokens | Mean latency | Wall time | Throughput |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GPT-5.4-mini | 0 | 578,624 | 20,716 | 0 | 599,340 | 2.708 s | 31.301 s | 6.390 req/s |
| GPT-4.1-mini | 0 | 578,824 | 21,363 | 0 | 600,187 | 3.133 s | 23.115 s | 8.652 req/s |

## Findings

1. Neither mini model is consistent enough with Luna to replace it.  Both
   heavily over-select anchors: 146 and 107 positives versus Luna's 25.
2. GPT-4.1-mini is materially closer than GPT-5.4-mini, but its 55% agreement,
   19.6% precision, and kappa 0.145 remain poor.  On this 12.5%-positive set,
   even an always-negative rule would have 87.5% raw accuracy.
3. Their dominant failure is speculative continuation from a single token.
   They routinely convert ordinary lexical differences such as `Understand`
   versus `Param`, `se` versus `sequential`, or `scalar` versus `determinant`
   into a reasoning branch or correction when Luna marks the evidence as
   insufficient, surface-only, or same-reasoning.
4. Both endpoints are operationally attractive—zero failures, no reasoning
   tokens, and about 600k total tokens per 200 judgments—but low quality, not
   serving cost, is the blocking issue.

## Decision

Do not run balanced held-out or the full trace pipeline with either model under
the current single-token rubric.  If a cheaper judge is still needed, first add
student/teacher continuation lookahead and then re-evaluate GPT-4.1-mini; it is
the less-bad of these two candidates.
