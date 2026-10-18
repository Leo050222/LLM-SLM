# Original-context student resampling at threshold 2

Job 107861, one node / eight H800, 64 CPU / 512 GiB, four student/teacher pairs, 128 concurrent traces. Eight datasets / 4571 rows, one rollout per row, same frozen inputs and R1 seed 20260912. Qwen3-1.7B student, Qwen3-14B score-only teacher. Earliest signed logP_student(token) minus logP_teacher(token) >2 triggers replacement of exactly one token. No anchor judge, no intervention cap.

Generation and replacement both use the original task prompt. At a trigger, recompute student logits on exactly original prompt token IDs + accepted response + draft before trigger; assert exact equality with handoff prefix. Mask the original trigger token and sample with temperature/top_p=1/1. Discard remaining draft and resume ordinary student generation. No corrective instruction, Existing reasoning or New reasoning sections are added. This reruns the identical prefix rather than caching the earlier full logits tensor. Replacement RNG salt is kept identical to the reprompt experiments to avoid an extra RNG-policy change.

Draft 64, output/context limits 40264/40960, cumulative trace timeout 2000 seconds. This differs from the already-scored no-draft reprompt run's 900-second limit and should be noted in comparisons. Each intervention stores original draft, trigger gap, sampled token/logprob, seed and original-context hash. Nine CPU tests passed locally and remotely, including strict threshold, exact context, original-token rejection, EOS, timeout-independent context boundary, failed-call atomicity and resume. All four service and forced-trigger GPU smokes gate formal dispatch. Initial state PENDING Priority; GPU validation awaits allocation. Generation only.

Output: `/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_ablation/weak_resample_t2_k1_20260920`.

Recovery: job 107861 could not start teacher replica 3 on gnho008 GPU7 (No CUDA GPUs are available). No formal rollout was dispatched. Excluded gnho008 and added worker-liveness checks during smokes. Replacement job 107874; same inputs, checkpoints, cumulative 2000s deadline and scientific configuration.
