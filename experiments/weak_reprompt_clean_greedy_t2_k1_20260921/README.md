Full corrected Existing reasoning prompt rerun with greedy drafts and greedy banned-token replacement.
4571 frozen prompts, 8 datasets, one rollout each; 8 H800 GPUs, 4 student/teacher pairs.
Strict raw logprob gap >2, one replacement token, original token banned, per-trace cumulative deadline2000s.
Prompt and state machine unchanged from107927. Independent outputs; no results reused.
Slurm walltime2h to allow backfill; checkpoint continuation needed if job expires before completion.
GPU service and forced-intervention smokes gate all generation.
