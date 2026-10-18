# strong4_weak17_reprompt_greedy_t2_k1_20261005

Strong-main prompt repair experiment, greedy decoding, threshold 2, k=1, one 8-H800 node.

- Main and repair model: `Qwen3-4B`
- Weak comparison model: `Qwen3-1.7B`
- Generation: temperature 0, top_p 1, top_k -1
- Trigger: earliest response token with `log P(token | strong) - log P(token | weak) > 2`
- Intervention: modify the strong model prompt with the previous-continuation repair template, issue a fresh strong request, and ban the original strong token at the first position
- Input: 4,631 frozen prompts: code (2,317), math (2,254), AIME24 (30), AIME25 (30)
- Every bifurcation event stores both models' top-100 logprobs and token IDs/text, the original strong sampled token, and the replacement token
- Four strong/weak pairs per 8-GPU node, 32 concurrent traces per pair, 900 s cumulative per-trace deadline

The frozen input SHA256 is `b45f1290d23ecb333512b1b57b985024e559b19dd3603f4bade2fb52bc20d877`.
