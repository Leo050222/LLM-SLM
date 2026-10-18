# Refinement Report

**Problem**：从 token-local OPD 转向 outcome-calibrated corrective spans。  
**Date**：2026-08-12  
**Rounds**：2  
**Final Score**：7.25 / 10  
**Final Verdict**：REVISE

## Final Thesis

- 方法不是 fully on-policy；它保留 on-policy student roll-in，但 bridge 内采用 counterfactual teacher supervision。
- 真正可防御的贡献不是 teacher chunk，而是 prefix-specific recovery horizon。
- external intervention utility 与 parameter-update utility 必须分开验证。
- oracle horizon 与 cheap adaptive horizon、same-prefix writability 与 held-out generalization 必须分别报告。

## Score Evolution

| Round | Problem Fidelity | Method Specificity | Contribution Quality | Frontier Leverage | Feasibility | Validation Focus | Venue Readiness | Overall | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 8 | 6 | 5 | 6 | 5 | 7 | 4 | 5.90 | RETHINK |
| 2 | 9 | 8 | 6 | 7 | 7 | 8 | 5 | 7.25 | REVISE |

## Remaining Weaknesses

当前差距主要是实证而非方法复杂度：若 adaptive recovery horizon 不能在等预算下胜过 global fixed span，或 intervention effect 不能预测训练收益，则不应继续扩展系统。

## Output Files

- `refine-logs/FINAL_PROPOSAL.md`
- `refine-logs/REVIEW_SUMMARY.md`
- `refine-logs/round-1-review.md`
- `refine-logs/round-2-review.md`
- `refine-logs/score-history.md`
