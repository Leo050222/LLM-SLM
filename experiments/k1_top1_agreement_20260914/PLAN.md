# Top-1 agreement immediately after one teacher token

Job103878 k1_top1_agree completed0:0 ongnho038 in00:04:00. Both model
prefix witnesses passed with identical top1 and zero pmax delta. Results fetched
locally:3812 traces,32360 paired positions; overall agreement25140/32360.

Reuses the frozen100 questions per8 datasets and5 thresholds of the expanded
entropy sample; only k=1. No new stochastic rollouts. Uses every teacher bridge
with one retained token and a subsequent retained response position. Expected
3812 traces and32360 paired positions, verified during preparation.

Both models see identical prompt+response[:after_bridge] token IDs. The final
hidden state predicts the next token; raw untruncated logits at temperature1,
without an active first-token ban, define argmax and softmax pmax. This is top-choice
agreement, not agreement of independent random samples. Record top-1 ID/text,
max probability, top1-top2 probability margin and entropy. Complete tokenizer
vocabulary identity and source hashes must match. BF16 forward/FP32 softmax.
Separate all-event, first-valid-intervention and trace-equal estimates; retain
dataset and threshold strata. No terminal next-token distribution after EOS.
Model-conditioned confidence is not calibrated correctness.

Warm verl-opd environment unchanged. One node gnho038, two H800s,16CPU,160GiB,
one-hour job limit, per-trace checkpoints. Local paired agreement fixture passed.
Runtime prefix witness verifies probability alignment; its argmax identity is
also logged because numerical near-ties can change argmax.
