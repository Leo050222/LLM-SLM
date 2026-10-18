# Weak-to-8B greedy prefix handoff

This run uses the frozen matched 4,571-question greedy set from the existing 1.7B and 4B baselines. For each weak-model rollout, the first floor(10%, 20%, 30%, 40%, or 50%) of response tokens is retained and Qwen3-8B greedily continues from the exact token prefix. The two arms are Qwen3-1.7B→Qwen3-8B and Qwen3-4B→Qwen3-8B. A Slurm allocation of two nodes and sixteen H800 workers is used.

Generation outputs are under `/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_prefix_handoff_weak_to_8b_20261005`. The local scripts are copied to the matching `opd_scripts` directory before submission. Prefix lengths, input hashes, model paths, sampling settings, and per-worker completion markers are recorded in the output directory.
