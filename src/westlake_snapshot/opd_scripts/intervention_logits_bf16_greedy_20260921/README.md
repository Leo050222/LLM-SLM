# Dual-greedy k=1, pi=2 intervention logits

Input: completed 4,571 main-method dual-greedy traces at `/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_ablation/main_greedy_k1_t2_20260921/r1_main_greedy_k1_t2`.

Output: `/storage/yukaichengLab/wangxin/test_eiloeo/opd_analysis/intervention_logits_bf16_greedy_20260921`.

Same archived-prefix protocol as stochastic x8: raw/unmasked full 151,936-dimensional logits from both Qwen3-1.7B and Qwen3-14B, BF16 safetensors, 512-token SDPA replay chunks, stable event IDs, source/prefix hashes, historical logprobs, no-intervention inventory and full file verification. Includes interventions in timeout traces. No new rollout generation. Raw softmax at T=1 describes model preference, not greedy policy sampling probabilities.

Environment: unchanged `.aris/compute/verl-opd-env.json` canonical SHA256 `5ae61e04e36b0afd56a5b580951741b3f4b16813c0a1acb521dfa0691c3e8107`; use `/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python` on the cluster. One normal-queue H800 per model, no fixed node. Existing gnho008 UUID workaround retained; unhealthy card rejected. No environment installation needed.

Non-launching invocation for independent documentation verification:

```bash
ssh -o BatchMode=yes gpu-wangxin 'cd /storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/intervention_logits_bf16_greedy_20260921 && /storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python -m unittest -v test_run.py'
```

Freeze with `python run.py prepare`; submit gpu.slurm separately with student/teacher arguments and verify.slurm after both succeed. The Slurm GPU payload runs seeded CUDA witness, tokenizer equivalence, BF16 bitwise serialization, fixed-prefix smoke from all datasets, longest-context replay, and chunk/full-prefill alignment before full archive. Check LAUNCH.json before any submission. Final ALL_LOGITS_VERIFIED.json is the completion gate.

`replay_base.py` is the previous validated archive engine; its verifier uses manifest event count instead of the old x8 constant. `run.py` validates the new greedy source and builds independent frozen inputs. The original stochastic archive remains separate.

## Launch 2026-09-21

- Frozen: 4571 traces, 43879 intervention events, 117 zero-intervention traces, 98 timeout traces (existing interventions included), 44 shards per role; 2343360 replay input tokens per model.
- Tensor payload for both roles: 26667198976 bytes (26.67 GB; metadata and smoke files extra).
- Manifest SHA256: `06a3728d5adf10ddd32324d634e331cb8eab4a4990f3909ae3d9a60ec75ca4bb`.
- Student job108227, teacher job108228, dependent verifier108229. Normal queue allocated both single-GPU jobs to gnho008; no node was requested.
- Local and fresh-agent remote invocation: all8 causal-alignment tests passed. Both GPU witnesses and8-dataset per-role model smokes passed. Student historical logprob median/max absolute error0.03330/0.11227; teacher0.24990/0.74771. These are replay differences, not probability percentage points or full-data error bounds.
- Final verification108229 COMPLETED 0:0 in22s. Both model jobs COMPLETED 0:0 (student3m23s,teacher5m28s). ALL_LOGITS_VERIFIED.json confirms43879events per model,44shards per model,all checksums/types/shapes/event IDs passed. Final shard bytes26667951024 combined.
