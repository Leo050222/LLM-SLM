#!/bin/bash
#SBATCH -J math_grid_coloc16
#SBATCH -p yukaichenglab
#SBATCH --account=yukaicheng
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:H800:8
#SBATCH --exclude=gnho005,gnho009,gnho018
#SBATCH --mem=768G
#SBATCH --time=UNLIMITED
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/math_grid_coloc16.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/math_grid_coloc16.%j.err
set -Eeuo pipefail
export PYTHONDONTWRITEBYTECODE=1
export OPD_CODE_ROOT=/storage/yukaichengLab/wangxin/test_eiloeo/opd_online_code/20260911_chunk64_k8_2nodes
export GRID_ROOT=/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_hard/math_reasoning_grid_x8_20260916
export SCALE_CODE=/storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/grid_math_colocated16_20260918
export OPD_PYTHON=/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python
export GRID_REPLICAS=8
export EXPECTED_ROWS=2254
export OPD_ONLINE_RUN=$GRID_ROOT/slots/job_$SLURM_JOB_ID
[[ ! -f "$GRID_ROOT/ALL_GENERATION_COMPLETE.json" ]] || exit 0
mkdir -p "$OPD_ONLINE_RUN/logs"
/soft/slurm/bin/srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --cpus-per-task=1 --kill-on-bad-exit=1 bash "$SCALE_CODE/preflight.sh"
"$OPD_PYTHON" "$SCALE_CODE/cutover.py"
nvidia-smi --query-gpu=index,name,memory.used --format=csv
/soft/slurm/bin/srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --cpus-per-task=64 --cpu-bind=cores --kill-on-bad-exit=1 bash "$SCALE_CODE/node.sh"
test -f "$GRID_ROOT/ALL_GENERATION_COMPLETE.json"
