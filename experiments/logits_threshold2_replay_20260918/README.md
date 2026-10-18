# Threshold-2 CRUXEval-I logits replay

Submitted job 106102: one H800, warm unchanged verl-opd environment.
Qwen3-1.7B student and Qwen3-14B teacher, BF16, Transformers SDPA.
This preserves the previous diagnostic backend, not a vLLM reproduction.

Uniform first sample with Python seed 20260918; if it finishes without an
anchor, inspect remaining samples in seeded shuffled order and preserve all
attempts in search_history.json. Native non-thinking prompts from the existing
CRUXEval-I grid. Temperature 1, full-vocabulary student sampling, same seed.

Student generates incremental drafts of up to 64 tokens, saving every raw
next-token logit vector before sampling. Teacher scores each student's draft
with full-prefix teacher forcing. Select the earliest position with strictly
log P_student(sampled token) - log P_teacher(same token) > 2, matching the
signed rule used in code_grid_x8 and nojudge_grid. No external judge.

The saved prefix EXCLUDES the trigger token. Its saved original logits predict
that token. Replay the serialized IDs without re-tokenizing and without the
original KV cache. Repeat prefill and incremental teacher-forced execution
as controls. Save all four full-vocabulary tensors, source row, prefix IDs,
all threshold checks up to the first trigger, and top8/top16 comparisons.
Probability comparisons use FP64 softmax of saved BF16 logits; threshold
selection uses FP32 log_softmax, matching the earlier diagnostic sampling.
Top-k reports break ties by ascending token ID and include cutoff tie sets.

No env rebuild, no modification of existing experiments or shared weights.
Remote results: opd_profile/logits_threshold2_replay_20260918/106102.
