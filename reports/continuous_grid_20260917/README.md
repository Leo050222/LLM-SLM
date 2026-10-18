# Continuous grid allocation, 2026-09-17

User explicitly requested holding the allocated GPUs until completion instead of six-hour segments. Slurm yukaichenglab MaxTime=UNLIMITED and normal QoS has no MaxWall. Updated existing pending jobs105262(code) and105320(math/reasoning) in place to TimeLimit=UNLIMITED; no cancellation/resubmission.

Both node.sh launchers now keep the same eight workers resident while a coordinator invokes successive dispatcher passes within one allocation. An internal five-hour dispatcher pass boundary may remain, but no worker exit, allocation release, or queue re-entry occurs at that boundary. ALLOCATION_DONE is written only after GRID_ROOT/ALL_GENERATION_COMPLETE.json exists. Real errors fail the job rather than silently marking completion. Local future batch scripts also requestUNLIMITED and assert full completion instead of submitting successors. The already queued Slurm batch snapshots call these updated node.sh paths; their old successor conditional is unreachable on successful partial passes, since node.sh returns successfully only after full completion.

Python files and scientific manifests were unchanged; threshold/k/seed settings, per-trace cumulative900s rules, and existing results/checkpoints are preserved. Bash syntax passed. A CPU-only mock ran three dispatcher passes with the same eight worker processes and released workers only after full completion. Test: test_allocation.py.

Remote deployment audit/backups: /storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/continuous_grid_20260917.
