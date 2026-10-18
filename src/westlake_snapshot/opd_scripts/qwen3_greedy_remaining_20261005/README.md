# Qwen3 greedy completion of the math/code benchmark set

This run fills the six datasets absent from `qwen3_greedy_hard_three_20261005`:
MATH500 (500), CRUXEval-O (800), CRUXEval-I (800), HumanEval+ (164), MBPP+
(378), and LiveCodeBench (175), for 2,817 prompts per model. Together with
AIME24/AIME25, OlympiadBench, and BBH from the hard-three run, each model has
10 datasets and 4,631 scored prompts.

Both Slurm jobs request one node with 8 H800 GPUs. Generation is Qwen3 vLLM
completion with `temperature=0`, `top_p=1`, `n=1`, seed `20261005`, and a
40,960-token context. Math uses the existing evalscope symbolic-equivalence
parser; code uses the existing CRUXEval, EvalPlus, and LiveCodeBench harnesses
inside bubblewrap.
