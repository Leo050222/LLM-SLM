#!/bin/bash
set -Eeuo pipefail
P=/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_hard/nojudge_grid_20260912
S=/storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/nojudge_grid_20260912
exec 9>"$P/score_supervisor.lock"
flock -n 9 || exit 0
while [[ ! -f "$P/ALL_GENERATION_COMPLETE.json" ]]; do sleep 30; done
if [[ ! -f "$P/score_job.txt" ]]; then
 bash -ilc "/soft/slurm/bin/sbatch --parsable $S/score.slurm" >"$P/score_submission.log" 2>&1
 /storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python - "$P" <<'PY'
from pathlib import Path
import sys,re
p=Path(sys.argv[1]);s=(p/'score_submission.log').read_text();lines=[l.strip() for l in s.splitlines() if re.fullmatch(r'\d+(;[^\s]+)?',l.strip())];assert len(lines)==1;(p/'score_job.txt').write_text(lines[0].split(';')[0])
PY
fi
