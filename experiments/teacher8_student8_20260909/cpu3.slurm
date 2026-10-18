#!/bin/bash
#SBATCH --job-name=t8s8_score_3nodes
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=3
#SBATCH --ntasks=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=512G
#SBATCH --time=12:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/t8s8_score3.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/t8s8_score3.%j.err
set -Eeuo pipefail
trap 'rc=$?; echo "BATCH_ERROR rc=$rc line=$LINENO command=$BASH_COMMAND" >&2' ERR
source /storage/yukaichengLab/wangxin/miniconda3/etc/profile.d/conda.sh
conda activate verl-opd
export ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
export OUT=${ROOT}/opd_eval_deepmath/teacher8_student8_20260909
export PYTHONPATH=${ROOT}/opd_scripts:${ROOT}/verl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
export PYTHONUNBUFFERED=1
/soft/slurm/bin/scontrol show job "$SLURM_JOB_ID"
/soft/slurm/bin/srun --nodes=3 --ntasks=3 --ntasks-per-node=1 --cpus-per-task=96 --cpu-bind=cores --kill-on-bad-exit=1 --output="${ROOT}/opd_logs/t8s8_score3.${SLURM_JOB_ID}.rank%t.log" --error="${ROOT}/opd_logs/t8s8_score3.${SLURM_JOB_ID}.rank%t.err" bash -c 'exec python "$ROOT/opd_scripts/teacher8_student8_20260909/score_shard.py" --output "$OUT" --rank "$SLURM_PROCID" --shards 3 --workers 80'
python - <<'PY'
import json,os
from pathlib import Path
p=Path(os.environ['OUT'])/'score_shards'
rows=[json.loads((p/f'rank{i}.json').read_text()) for i in range(3)]
assert sum(x['total'] for x in rows)==33000
assert all(x['job_id']==os.environ['SLURM_JOB_ID'] for x in rows)
print('ALL_SCORE_SHARDS_COMPLETE',flush=True)
PY
# Original scorer validates all cached hashes and performs the existing final merge.
/soft/slurm/bin/srun --nodes=1 --ntasks=1 --cpus-per-task=96 python "${ROOT}/opd_scripts/teacher8_student8_20260909/score.py" --output "$OUT" --workers 1
echo SCORING_ALL_COMPLETE
