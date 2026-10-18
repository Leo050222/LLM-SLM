#!/bin/bash
#SBATCH -J nojudge_grid
#SBATCH -p yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:H800:4
#SBATCH --exclude=gnho009,gnho018
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/nojudge_grid.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/nojudge_grid.%j.err
set -Eeuo pipefail
export PYTHONDONTWRITEBYTECODE=1
export OPD_CODE_ROOT=/storage/yukaichengLab/wangxin/test_eiloeo/opd_online_code/20260911_chunk64_k8_2nodes
export GRID_ROOT=/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_hard/nojudge_grid_20260912
export OPD_ONLINE_RUN=$GRID_ROOT/slots/slot_$GRID_SLOT
mkdir -p "$OPD_ONLINE_RUN/logs"
export OPD_PYTHON=/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python
/soft/slurm/bin/srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --cpus-per-task=32 --cpu-bind=cores --kill-on-bad-exit=1 bash /storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/nojudge_grid_4gpu_20260912/node.sh
