# OPD rollout token heatmap

This viewer packages a completed OPD token diagnostic into one offline HTML file.
It supports rollout selection, Forward KL, sampled reverse-KL k1/k3, token-window
navigation, per-token heat coloring, and hover/click details.

## Export a page

The input can be either a diagnostic directory containing `tokens.parquet` or
the parquet file itself:

```bash
python scripts/export_opd_token_heatmap.py \
  /path/to/aime24_step0_t1_full_seed20260806 \
  --output reports/aime24_rollout_token_heatmap.html \
  --title "AIME24 rollout token KL heatmap"
```

Open the generated HTML directly in a modern browser. It is self-contained and
does not fetch data or JavaScript from the network.

## Required parquet columns

- `dataset_index`
- `problem_id`
- `correct`
- `response_position`
- `token_text`
- `raw_forward_kl_top128`
- `chosen_logprob_gap`

The exporter derives reverse-KL estimators from the chosen-token log-probability
gap:

- `k1 = log pS - log pT = -chosen_logprob_gap`
- `k3 = exp(chosen_logprob_gap) - 1 - chosen_logprob_gap`

Token text is preserved losslessly. Per-token values are stored with 8-bit
logarithmic quantization for a compact standalone page; rollout summary values
are embedded at their original floating-point precision.
