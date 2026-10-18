#!/bin/bash
#SBATCH -J code_grid
#SBATCH -p yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:H800:4
#SBATCH --exclude=gnho009,gnho018
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/code_grid.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/code_grid.%j.err
set -Eeuo pipefail
export PYTHONDONTWRITEBYTECODE=1
export OPD_CODE_ROOT=/storage/yukaichengLab/wangxin/test_eiloeo/opd_online_code/20260911_chunk64_k8_2nodes
export GRID_ROOT=/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_code/code_reasoning_grid_20260913
export OPD_ONLINE_RUN=$GRID_ROOT/slots/job_$SLURM_JOB_ID
mkdir -p "$OPD_ONLINE_RUN/logs"
export OPD_PYTHON=/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python
/soft/slurm/bin/srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --cpus-per-task=32 --cpu-bind=cores --kill-on-bad-exit=1 bash /storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/code_reasoning_grid_20260913/node.sh

if [[ ! -f "$GRID_ROOT/ALL_GENERATION_COMPLETE.json" ]]; then
 /soft/slurm/bin/sbatch --parsable --dependency=afterok:$SLURM_JOB_ID /storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/code_reasoning_grid_20260913/gpu.slurm >> "$GRID_ROOT/job_ids.txt"
fi
