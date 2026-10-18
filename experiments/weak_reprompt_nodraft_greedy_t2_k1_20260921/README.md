# weak_reprompt_nodraft_greedy_t2_k1_20260921

All 4571 frozen prompts, eight datasets, one rollout per question.
4 H800 GPUs, 2 student/teacher pairs, concurrency 64.
Draft and replacement greedy (temperature 0); original replacement token banned.
Raw logprob threshold strictly >2; score temperature 1.
Unchanged prompt and state machine from weak_reprompt_nodraft_t2_k1_20260920.
Cumulative trace deadline 2000 seconds; timeout counts wrong.
Independent output and queue directories; no stochastic outputs reused.
