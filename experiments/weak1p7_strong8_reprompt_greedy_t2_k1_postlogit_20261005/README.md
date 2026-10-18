# weak1p7_strong8_reprompt_greedy_t2_k1_20261005

Reverse prompt-repair experiment on 4,631 frozen prompts. Qwen3-1.7B is the main generator and Qwen3-8B is the intervention/scoring model. Both use greedy decoding (`temperature=0`, `top_p=1`, `top_k=-1`).

At the earliest response token where `log P(1.7B sampled token | prefix) - log P(Qwen3-8B sampled token | prefix) > 2`, the accepted prefix is kept, a fresh repair request is built with the same previous-continuation template, and Qwen3-1.7B generates exactly one replacement token with the original 1.7B token banned. Generation then continues from the replacement.

Every intervention event stores the main 1.7B sampled token, replacement token, and the main/intervention models' top-100 token IDs/text/logits at the trigger position. Input has code (2,317), math/BBH/Olympiad (2,254), AIME2024 (30), and AIME2025 (30). One Slurm node with 8 H800 GPUs is used, with four student/intervention pairs and 32 concurrent traces per pair.
