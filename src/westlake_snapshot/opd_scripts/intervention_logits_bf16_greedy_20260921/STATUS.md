# Greedy logits archive status

Source: main dual-greedy k=1, pi=2 (4571 traces, one rollout per question).

| Dataset | Traces | Interventions |
|---|---:|---:|
| CRUXEval-O | 800 | 1977 |
| CRUXEval-I | 800 | 3223 |
| HumanEval+ | 164 | 945 |
| MBPP+ | 378 | 2094 |
| LiveCodeBench | 175 | 4133 |
| OlympiadBench | 674 | 8589 |
| BBH | 1080 | 19181 |
| MATH500 | 500 | 3737 |
| Total | 4571 | 43879 |

44 paired shards; 117 traces have no intervention; 98 generation-timeout traces retained. Longest replay prefix10404 tokens. Tensor payload26.67GB combined. Values are raw unmasked BF16 logits, all151936 dimensions.

Jobs: student108227,teacher108228,verifier108229. **All COMPLETED 0:0**; elapsed3m23s/5m28s/22s. `ALL_LOGITS_VERIFIED.json` confirms all88 archives passed checksum, BF16,shape,event-ID coverage and pairing checks. Each model43879events/44shards/13333975512bytes; combined26667951024bytes (26.67GB, including headers and event IDs; excludes smoke/inputs).

Load one event:

```python
from safetensors import safe_open
root = '/storage/yukaichengLab/wangxin/test_eiloeo/opd_analysis/intervention_logits_bf16_greedy_20260921'
with safe_open(root + '/student/00000.safetensors', framework='pt', device='cpu') as f:
    event_ids = f.get_tensor('event_ids')
    logits = f.get_slice('logits')[0]  # BF16 raw logits
    preference_logprobs = logits.float().log_softmax(-1)  # T=1, not greedy action probabilities
```

The matching teacher file has exactly the same row/event IDs. Source metadata and original/replacement token IDs are in `inputs/00000.jsonl`, flattening each row's events in order. Historical source and prefix hashes preserve identity. The historical greedy policy chooses argmax (teacher first bans original student token); softmax T=1 is for distribution analysis.
