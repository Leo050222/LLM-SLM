#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH -J online_k8_score
set -euo pipefail
: "${OPD_CODE_ROOT:?}"
: "${OPD_ONLINE_RUN:?}"
: "${OPD_PYTHON:?}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
export PYTHONPATH=/storage/yukaichengLab/wangxin/test_eiloeo/verl:${PYTHONPATH:-}
if [[ ! -f "$OPD_ONLINE_RUN/GENERATION_COMPLETE" ]]; then
  echo 'Generation did not complete; refusing partial-cohort scoring' >&2
  exit 1
fi
"$OPD_PYTHON" "$OPD_CODE_ROOT/ocrd_pipeline/online_rollout/evaluate.py" --run-dir "$OPD_ONLINE_RUN/results"
touch "$OPD_ONLINE_RUN/EVALUATION_COMPLETE"
