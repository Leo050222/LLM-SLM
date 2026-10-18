# CRUXEval-I token-six raw logits replay

Run: `run.py` through `gpu.slurm`; submitted Slurm job 106101.
Uses the unchanged warm verl-opd environment, local Qwen3-1.7B weights,
Transformers SDPA, BF16, eval/inference mode, one H800, batch size one.
No environment rebuild. Existing code-grid non-thinking prompts are reused.
A single CRUXEval-I row is sampled with Python seed 20260918.

The first forward prefills the original prompt; five tokens are sampled with
Torch seed 20260918, temperature 1 and full vocabulary. After each selection
we feed that token through the original KV cache. Consequently the last
forward predicts token six, although only five tokens have been sampled.
The full original prompt plus five output IDs is serialized and read back
unchanged for a separate fresh prefill, without passing the original cache.
Raw full-vocabulary logits are captured before softmax/sampling processing.
Controls repeat both fresh-prefill and incremental teacher-forced execution.
The primary comparison intentionally changes the execution shape from
incremental decode to full prefill. Exact agreement is measured, not assumed.

Artifacts: sample.json, prefix_token_ids.json, result.json, and four `.pt`
raw logits tensors. Result records versions, selected tokens, exact/bitwise
agreement, differing positions, max/mean/RMS absolute differences, argmax,
and maximum softmax probability difference. This is a single example and
backend-specific diagnostic, not evidence about all prompts or vLLM.

Remote outputs:
`/storage/yukaichengLab/wangxin/test_eiloeo/opd_profile/logits_replay_20260918/106101/`
