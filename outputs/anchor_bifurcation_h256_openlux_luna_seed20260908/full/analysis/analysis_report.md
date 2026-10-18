# 256-token bridge anchor bifurcation analysis

## Scope

- 900 distinct question–bridge groups: 100 groups in each empirical success-rate-lift bin from 0/8 to 8/8.
- One 256-token bridge per selected question.
- Eight blinded direct-vs-bridge trajectory-pair judgments per group (7,200 judgments total).
- Judge: `gpt-5.6-luna` through OpenLux.
- Primary endpoint: `anchor_verdict == yes` under the strict logical-bifurcation definition.
- Question-level endpoint: at least 5 of 8 pair judgments are `yes`.

## Raw results by empirical success-rate lift

| Lift | Pair-level anchor=yes | 95% cluster-bootstrap CI | Majority yes questions | Unanimous yes questions | Immediate divergence | Direction changed | Persistent |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0/8 (0pp) | 493/800 (61.6%) | 54.9–68.1% | 60/100 (60%) | 27/100 (27%) | 70.9% | 86.2% | 84.4% |
| 1/8 (12.5pp) | 421/800 (52.6%) | 46.1–59.1% | 50/100 (50%) | 15/100 (15%) | 57.6% | 85.9% | 85.9% |
| 2/8 (25pp) | 458/800 (57.2%) | 50.5–63.9% | 54/100 (54%) | 21/100 (21%) | 62.3% | 87.6% | 84.4% |
| 3/8 (37.5pp) | 493/800 (61.6%) | 54.8–68.1% | 62/100 (62%) | 24/100 (24%) | 65.4% | 91.6% | 87.6% |
| 4/8 (50pp) | 504/800 (63.0%) | 56.0–69.6% | 61/100 (61%) | 27/100 (27%) | 67.8% | 89.2% | 84.4% |
| 5/8 (62.5pp) | 579/800 (72.4%) | 66.5–78.0% | 69/100 (69%) | 36/100 (36%) | 77.4% | 91.9% | 86.8% |
| 6/8 (75pp) | 539/800 (67.4%) | 61.1–73.4% | 66/100 (66%) | 26/100 (26%) | 71.8% | 87.9% | 85.9% |
| 7/8 (87.5pp) | 572/800 (71.5%) | 65.2–77.4% | 72/100 (72%) | 40/100 (40%) | 76.6% | 88.9% | 85.2% |
| 8/8 (100pp) | 619/800 (77.4%) | 70.9–83.4% | 79/100 (79%) | 51/100 (51%) | 80.2% | 88.8% | 89.0% |

## Main findings

1. The intended trend is present but not strictly monotonic. A linear trend fitted to independent question-level means increases by 2.51 percentage points in `anchor=yes` per +1/8 success lift (cluster bootstrap 95% CI: +1.68 to +3.35pp; normal-approximation p=3.4e-9). Question-level Pearson r is 0.194, so lift is informative but far from sufficient to identify a bifurcation.
2. The 8/8-lift endpoint is materially stronger than the 0/8 bin. Pair-level `anchor=yes` rises from 61.6% to 77.4%, a +15.8pp difference (cluster-bootstrap 95% CI: +6.6 to +24.9pp). Majority-yes questions rise from 60% to 79%, a +19pp difference (95% CI: +7 to +31pp). Unanimous-yes questions rise from 27% to 51%.
3. Coarser binning is cleaner: low lift (0–2/8) has 57.2% pair-level yes, 54.7% majority yes, and 21.0% unanimous yes; medium lift (3–5/8) has 65.7%, 64.0%, and 29.0%; high lift (6–8/8) has 72.1%, 72.3%, and 39.0%.
4. The component that tracks lift most visibly is precise divergence timing. Immediate-after-anchor divergence rises from 70.9% at 0/8 to 80.2% at 8/8. Direction change and persistence are high in every bin (roughly 86–92% and 84–89%), so they distinguish useful bridges less strongly.
5. Outcome-cell results are consistent with successful redirection: `F_to_S` pairs have the highest strict-bifurcation rate (73.0%), followed by `F_to_F` (66.5%), `S_to_F` (57.0%), and `S_to_S` (50.4%). These are diagnostic rates because the eight pairs were intentionally cell-stratified, not sampled proportionally from all 64 possible pairings.
6. The most common labeled mechanisms are changed strategy (5,357/7,200; 74.4%) and changed interpretation (3,851/7,200; 53.5%). Multiple labels are permitted. Changed assumption, new subgoal, and error correction occur in 19.3%, 19.4%, and 16.4% of judgments, respectively.
7. Blinded path order is well balanced and shows negligible bias: direct-as-A has 65.2% `anchor=yes`, versus 64.7% when direct is B (difference +0.49pp).

## Interpretation

The experiment supports the proposed association: bridges with larger empirical improvements are more often judged to begin at a genuine logical bifurcation, and the association becomes clearest at high lift. However, correctness lift alone is not a clean bridge-quality oracle. Even zero-lift bridges receive a strict-bifurcation verdict in 61.6% of pair judgments, showing that changing reasoning direction is common without improving success.

The most defensible selection rule from this experiment is therefore a conjunction: empirical lift plus stable semantic evidence. For example, prioritize bridges with positive lift, majority or unanimous `anchor=yes`, immediate divergence, and a persistent changed-strategy/changed-interpretation label.

## Limitations

- The eight observations per bridge are different trajectory-pair comparisons, not eight independent judge models; uncertainty is therefore clustered by question–bridge group.
- Lift is estimated from only eight direct and eight bridge rollouts. Extreme bins, especially 8/8, are subject to selection noise and regression to the mean.
- The 8/8 bin necessarily contains only `F_to_S` pairs because direct success is 0/8 and bridge success is 8/8. Its endpoint cannot fully separate lift from paired-outcome composition.
- Different questions appear in different lift bins. Topic and mean difficulty appear broadly similar, but unmeasured question-composition effects remain possible.
- There is no placebo-anchor or random-bridge negative control, and all judgments come from one judge model.

## Recommended next analyses

1. Re-rollout a held-out subset of bridges to estimate lift out of sample, especially the 0/8 and 8/8 bins.
2. Add matched placebo anchors and randomly selected 256-token spans to measure the judge's base positive rate.
3. Rejudge a stratified subset with a second model or human annotators and report inter-rater agreement.
4. Test whether the conjunction of lift, majority bifurcation, immediate onset, and persistent strategic change predicts held-out rollout accuracy better than lift alone.
