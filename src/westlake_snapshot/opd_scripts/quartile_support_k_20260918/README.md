# Fixed-prefix quartile support-length experiment

Eight existing datasets, 100 distinct source rollouts per dataset sampled uniformly without filtering correctness or observed intervention events, seed 2026091807. Sources are the original r1_k1_t2 completions. Randomly ordered samples are assigned 25 per response-token quartile; one position is drawn uniformly from each sample's assigned interval [floor(q*L/4), floor((q+1)*L/4)). Prompt tokens do not count. Prefix preserves exact source tokens, including any prior teacher tokens. Source file hashes, finish reasons, sampled IDs, and positions are frozen before submission. Timeout-censored source rollouts remain explicitly identified in sampling_audit.json.

For each of 800 prefixes, request exactly 32 new Qwen3-14B tokens ONCE (temperature=1, top_p=1, no first-token ban). Save this bridge independently. k=1,4,8,16,32 use slices of this identical saved sequence. Each sliced branch continues with Qwen3-1.7B at temperature=0 until EOS or the existing response/model token limits of 40264/40960. No further teacher interventions. Teacher early EOS terminates only branches that include the EOS endpoint; shorter slices continue with the student. All sampled prefixes must fit the 32-token request, checked before submission. Total: 800 teacher bridges and 4000 final responses.

Single node, 8 H800, 4 resident student/teacher pairs on separate GPUs, 64 concurrent samples, 64 CPUs, 512 GiB, unlimited Slurm time. Reuses existing verl-opd environment without dependency changes. GPU workers execute seeded CUDA witness and per-pair generation smoke including greedy seed invariance before dispatch. Checkpoints and completed outputs are hash-verified and resumable. Infrastructure lease failures preserve checkpoints and fail the job rather than marking answers incorrect.

Dependent CPU scoring uses existing sandboxed benchmark tests for five code datasets and existing local rule-based graders for BBH, Math500 and OlympiadBench. This is not GPT-4o-mini adjudication; compare k within this experiment using the same scorer. Reports overall 100-sample accuracy and each 25-sample quartile per dataset/k, plus paired wrong-to-correct and correct-to-wrong counts against k=1. Scoring infrastructure errors are retried and must resolve before the completion marker is written.

Remote code: /storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/quartile_support_k_20260918
Remote outputs: /storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_prefix/quartile_support_k_20260918
Results: summary.json and summary.csv. Job receipt: LAUNCH.json.

Validation: five semantic tests cover one shared bridge, exact slices and greedy continuation, early EOS, checkpoint resume, quartile boundaries and mismatched checkpoints. Python compilation and Bash syntax checks passed. Remote preflight additionally checks all source prefixes and real scoring adapters.
