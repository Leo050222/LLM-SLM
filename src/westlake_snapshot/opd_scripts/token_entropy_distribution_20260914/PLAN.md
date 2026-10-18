# Expanded full-token entropy distributions

User requested gnho020, two H800 GPUs. Use warm unchanged verl-opd environment,
one model per GPU,16 CPUs,160GiB,12h scheduler limit; no automatic resubmission.
Read-only replay of frozen outputs, not new stochastic generation.

100 deterministic questions per each of8 datasets, all25 k/threshold configs,
20000 traces. Same seed/order as earlier20-per-dataset sample, a nested expansion.
No filtering by correctness, interventions, length or timeout. AIME excluded.
One distribution value per retained response token, including EOS if present,
conditioned on the exact shared prefix before that token. Both Qwen3 models are
non-thinking; full vocabulary raw logits softmax at T=1, no truncation or banning
during measurement. Earlier first-token generation ban remains in source traces.

BF16 model forward, FP32 entropy in nats. Chunk64 vocabulary projections prevent
full sequence x vocabulary allocations. Validate frozen source hashes, tokenizer
identity, entropy bounds and3 prefix alignment positions on the first trace.
Store exact per-token FP32 entropies in model-specific NPZ; resume per trace.

Outputs: histograms and CDF inputs for both models, full-token and trace-equal
weightings, per dataset/config and global mixtures, split by student/teacher
generated tokens; mean, P10/25/50/75/90/95/99, H>=1/2/4 fraction and paired Hs>Ht.
Histogram bins have width.02nats; histogram-derived quantiles have at most.01nat
binning error relative to their empirical inverse-CDF position. Means and ratios
are computed directly from per-token values. Global mixtures are descriptive;
tokens/questions/configurations are dependent, so no naive token-binomial CIs.
Also retain every valid intervention's before/after entropy, not just two events.
Timeout partial outputs are included; length and historical stopping-rule effects
must be described with results. Terminal after-EOS distributions are not measured.

Launch preparation: distribution.py prepare --out <root>/opd_analysis/token_entropy_distribution_20260914
Then sbatch run.slurm. Completion requires paired model markers and COMPLETE.json.
