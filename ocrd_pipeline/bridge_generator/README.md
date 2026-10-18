# OCRD teacher bridge generator

This is stage 2 of the OCRD pipeline. It consumes an `ANCHOR_SELECTED`
artifact from stage 1 and generates exactly `h` tokens with the frozen teacher
model.

The input context is `anchor_prefix_token_ids`, which ends immediately before
the student's anchor token. Generation is split into two requests to one
loaded vLLM engine:

1. decode the student anchor token to text, verify that vLLM's single-token
   `bad_words` mapping contains the exact token ID, and generate one token with
   that hard blacklist;
2. generate the remaining `h - 1` tokens normally from the teacher, conditioned
   on the prefix and alternative first token.

This guarantees that the first bridge token differs from the student token
without banning that token from later bridge positions. `min_tokens` and
`max_tokens` are both set to the requested length for each phase and EOS is
ignored, so the artifact always contains exactly `h` tokens or the command
fails without writing a success artifact.

## Example

```bash
python run_bridge_generator.py \
  --anchor /path/to/trace_338/selected_anchor.json \
  --output /path/to/trace_338/bridge_h32.json \
  --h 32 \
  --temperature 1.0 \
  --top-p 1.0 \
  --seed 20260822
```

The output records the bridge token IDs/text, per-token chosen logprobs,
sampling parameters, the strict first-token constraint, and
`bridge_prefix_token_ids = anchor_prefix_token_ids + bridge_token_ids` for the
stage-3 continuation test.
