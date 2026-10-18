# Four-quartile matched random intervention

Submitted Slurm job 106061 on 2026-09-18: one node, four H800 GPUs, 32 CPUs, 256 GiB, UNLIMITED allocation. Two resident Qwen3-1.7B / Qwen3-14B pairs, 64 concurrent traces. Warm verl-opd environment unchanged. Runtime performs seeded GPU kernel witnesses and service smoke before dispatch.

One rollout for each of all 4,571 samples in the eight datasets, paired to existing r1_k1_t2 source completions (2,317 code and 2,254 math/reasoning). Source manifests and file hashes frozen. 32 source trajectories are timeout-censored and explicitly marked; their saved length defines the quarters. No sample is dropped. Planned interventions total 49,507.

Boundaries are floor(i*L/4) for i=1,2,3,4, expressed as cumulative output token counts. Token positions are zero based; intervals are [start,end). Uniform independent per-sample sampling without replacement within each interval uses seed 2026091801 and stable sample identity. Planned per-quarter counts exactly match the source. Positions are frozen before generation. No threshold scoring occurs during the new rollout. Source seed 20260912 and original dataset indices are retained for generation seeds; teacher k=1 and first-token ban remain unchanged.

Normal EOS is retained; no extra interventions beyond the source length. Output records planned and actual per-quarter counts, positions and unrealized interventions. Existing 900-second cumulative per-trace timeout remains unchanged and explicitly marked. Allocation remains resident until all queued samples finish, with no five-hour segmentation. This job generates outputs; it does not submit external answer-scoring API calls.

Remote code: /storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/random_quartile_k1_20260918
Remote output: /storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_ablation/random_quartile_k1_20260918
Slurm logs: /storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/random_quartile_k1.106061.{log,err}
Frozen prompts SHA256: 1f01fd67b2405a66f241b939273d8ac1260f8de06ef83557d8b2e76be68b4562

Validation: four CPU tests passed locally and in remote warm environment (count/position boundaries, natural EOS, reproducible uneven/empty quarters, checkpoint resume). Python compilation and Bash syntax checks passed. GPU runtime witness is pending allocation.

## Recovery job 106158

Job 106061 failed after 1m46s, following successful model service smoke, before any sample completed. The reused migration helper required an unrelated recovery_4gpu_20260917/migration.json. Replaced it with strict new-experiment initialization and exact manifest validation on resume; mismatched settings/inputs and orphan checkpoints still fail closed. Six tests pass locally and remotely, including manifest creation, resume, and rejection of changed configuration. Frozen inputs hash unchanged. Verified no same-experiment active job, then submitted four-H800 job 106158. Periodic monitoring remains PAUSED.
