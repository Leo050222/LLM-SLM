# Weak reprompt without old draft

Independent follow-up to job 107636. Same eight datasets / 4571 prompts, one rollout each, R1 seed 20260912, strict weak-minus-strong logprob gap >2, earliest trigger, 64-token drafts, temperature/top_p=1/1, max output/context=40264/40960, 900-second cumulative trace budget. Qwen3-1.7B generates; Qwen3-14B only scores. Four model pairs on one node with eight H800, 64 CPUs, 512 GiB.

The English raw completion prompt omits the entire Existing reasoning section and all discarded draft tokens. It says the previous continuation has a problem, then ends with New reasoning followed by the exact accepted prefix token IDs. Sample exactly one weak-model token while masking the original trigger token, then resume in the original task context. Original draft is retained only in audit logs. Existing experiment is not modified.

See reprompt.py for the exact frozen template. Context overflow is explicitly recorded; no silent truncation. Generation only.

Submitted job 107647 on gnho035. Nine CPU tests passed locally and remotely, including old-draft invisibility and rejection of the banned token. GPU allocation preflight passed with all eight GPUs initially empty. Four service and forced-trigger GPU smokes gate formal dispatch.

Launch verified: all four service smokes and all four forced-trigger reprompt smokes passed; formal rollout is writing completed outputs (at least 32 observed).
