# Timeout diagnosis: 2026-09-20 snapshot

Read-only analysis; running protocols and jobs were not changed. Metrics are a live partial snapshot, not final cross-dataset rates.

## Evidence

| Group | Timeout traces | Strong repetition in last 2048 tokens | Median length | Median rounds | Median interventions | Median accepted/generated ratio |
|---|---:|---:|---:|---:|---:|---:|
| Existing reasoning, 107873 | 384 | 379 (98.7%) | 8139 | 134 | 7.5 | 98.3% |
| Original-context resample, 107874 | 19 | 18 (94.7%) | 15437 | 249 | 8 | 99.1% |

Repetition metric: at least one exact 32-token window occurring >=10 times in the last 2048 tokens; overlapping occurrences counted. It is diagnostic, not a validated production stopping rule. In both timeout groups the median number of interventions in the last 20 rounds is zero. Sample tails repeat BR<BR, <BR/>, 5<5<, 9999, A A A, or a short numeric sequence. The four sampled traces in each group still accept 64-token drafts per round: no evidence of a no-progress state-machine loop.

Cached last-round scores confirm threshold blindness inside repetition. Existing-reasoning sample 00000: signed gap ranges 4.77e-7 to 3.81e-6, zero positions >2; student first logp=0, teacher≈-5.96e-7. Original-context sample 00019: gap≈1.19e-7, zero triggers; student logp=0, teacher≈-1.19e-7. Both models condition on the already-degenerate history and assign nearly unit probability to continued repetition. Agreement is not evidence of correct reasoning.

56/384 Existing-reasoning timeouts and 1/19 original-context timeouts previously banned EOS. Thus EOS replacement is a possible contributor, but not the general explanation. Normal-stop outputs inspected also include 109 and 100 traces respectively with an earlier EOS ban; correlation alone does not establish causality.

Existing-reasoning sample 00000 has early replacement tokens ` <` and `BR`, followed by BR<BR repetition. The prompt includes `<BRANCH_POINT>` and the temporary request is raw completion rather than the original chat template. Prompt-marker copying / distribution mismatch is a plausible contributor, not established by a controlled ablation. Inserting a single token from this altered context back into the original context can also disturb syntax. Direct resampling has no such marker but can disturb format under mandatory original-token masking.

## Runtime amplification

Worker disables prefix caching. Each round generates a draft and separately rescoring the full prefix plus draft on both models requests prompt_logprobs=16, although only actual draft-token probabilities are consumed. Score calls additionally request one unused generated token. Replacement makes another student inference on the original or revised prefix. Workers synchronously batch eight requests of the same op and return after the batch; each replica hosts 32 active traces. File queues add serialization/polling overhead; its exact time share has not been measured.

Recent direct-resampling sample: student generation batch median .361s, queue wait 4.798s; student scoring 2.209s, queue wait 3.429s; teacher scoring 3.173s, queue wait 2.742s. Batch runtime is shared across its requests and cannot be summed as per-trace GPU time. Statistics are transient and not directly comparable to job averages. Long contexts repeatedly prefilled and scored amplify compute, memory traffic and waiting. Budget counts active wall time including queue waits, not dedicated GPU execution time.

The 2000s continuation processes selected previously timed-out traces, so its all-timeout early batches are selection-biased, not a 100% timeout rate for the original experiment.

## Proposed remedies, in order

1. Detect pathological token-periodic suffixes conservatively (e.g. period 1..32, covering >=256 recent tokens and persisting over multiple rounds), record repetition_abort separately and preserve the entire partial output. Validate false positives on normal completed code/math traces before activation. This bounds waste but does not recover accuracy; do not silently change the running experiment's stop rule.
2. For actual recovery, roll back before the detected loop and use a separately evaluated rescue policy (teacher continuation or explicit reconsideration), with strict rescue-count/token caps. Report rescue accuracy and cost separately. Merely raising timeout repeats the same process longer.
3. New controlled variants: protect EOS from mandatory replacement; protect output-format boundaries or distinguish format tokens from reasoning tokens; remove artificial marker copying opportunities while retaining the intended prompt information. These change scientific semantics and require matched comparisons, not silent fixes.
4. Original-logit engineering: have student scoring produce the full-vocabulary masked sample on device, using a deterministic per-position seed; select the sample at the first teacher-gap trigger. Transfer token IDs, not dense logits tensors. Avoid top-k-only renormalization, which changes the distribution. Generation logprobs may replace separate student scoring only after confirming they match untempered scoring, including EOS and processed-logit semantics.
5. Optimize actual-token logprob extraction / scoring only the new suffix; remove unused top16 and unused decode only after validating vLLM support. Introduce persistent KV/prefix reuse carefully: cached-prefix prompt-logprob availability must be checked. Teacher-forcing score alignment is mandatory.
6. Benchmark 8/16/32 active traces per replica with identical samples and token budgets; isolate long contexts and use token-budget-aware batching rather than blindly increasing concurrency. Record queue wait, prefill, decode and score time separately. Do not treat scheduling-dependent wall time as equal scientific compute.

Recommended first step: offline replay/validate the repetition detector on saved outputs and implement non-semantic scoring optimizations in a separate version, then run a small paired pilot before another full rollout. No changes were applied to running jobs during this analysis.
