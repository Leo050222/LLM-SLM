# Qwen3 4B/8B greedy hard_three baseline

Two independent single-node Slurm jobs run Qwen3-4B and Qwen3-8B with all 8
H800 GPUs via vLLM tensor parallelism. They use the frozen 1,814-row
`hard_three_20260911` prompt set: AIME2024 (30), AIME2025 (30), OlympiadBench
English math (674), and BBH (27 subsets × 40 = 1,080). Generation is greedy
(`temperature=0`, `top_p=1`, `n=1`) with the existing native non-thinking chat
prompt and the existing per-dataset scoring rules.
