# DeepMath teacher-bridge smoke test

Date: 2026-08-22

Teacher: frozen Qwen3-14B. Sampling: temperature 1.0, top-p 1.0,
top-k disabled. Smoke bridge length: `h=32`.

| Trace | Student anchor token | Teacher bridge first token | Exact length | Constraint verified | Slurm evidence |
|---:|---|---|---:|---|---|
| 338 | ` identity` | ` additive` | 32 | yes | `91936_0 COMPLETED 0:0` |
| 216 | `×` | ` +` | 32 | yes | `91937_1 COMPLETED 0:0` |
| 485 | ` +` | `,` | 32 | yes | `91937_2 COMPLETED 0:0` |

Generated bridge text:

- 338: ` additive inverse of each element. The question is about whether such a finite subring must also be a division ring.\n\nFirst, let me think about the properties of`
- 216: ` + 1 = 3, E_2 is 2×3 + 1 = 7, E_3 is 2×3×5`
- 485: `, v2} (since each vector is a sum of k=1 vectors. Wait, actually, the first vector would be v1, then the next`

The initial array `91926` exposed that vLLM 0.23 limits
`allowed_token_ids` to 1,024 entries. Tasks 0 and 1 failed before decoding and
task 2 was cancelled before GPU use. The repaired implementation uses vLLM's
single-token `bad_words` mask only on the first decode request, after verifying
that decode-to-text and bad-word tokenization includes the exact student token
ID. The remaining `h-1` tokens are generated in a second unrestricted request
using the same loaded teacher engine.
