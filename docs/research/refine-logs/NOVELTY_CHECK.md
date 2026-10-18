# Novelty Re-check: Outcome-Calibrated Recovery-Span Distillation

Latest report: [NOVELTY_CHECK_20260812_144716.md](NOVELTY_CHECK_20260812_144716.md)

Verdict: **PROCEED WITH CAUTION**  
Estimated novelty: **4.5/10**

The full pipeline was not found verbatim in one prior paper, but its components closely overlap P-ALIGN, Relay-OPD, FutureBridge-OPD, ADWIN, DASH, TOPD, and PW-OPSD. The strongest defensible novelty is not chunk-level supervision itself; it is the proposed test of whether a per-prefix, outcome-based minimum recovery span predicts which corrective training update will improve the student's later free-running behavior under a matched budget.

See the dated report for the claim matrix, closest-work comparison, positioning, and kill experiments.
