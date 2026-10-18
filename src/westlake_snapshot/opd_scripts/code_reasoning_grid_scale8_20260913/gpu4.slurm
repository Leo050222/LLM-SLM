#!/bin/bash
#SBATCH -J code_grid4
#SBATCH -p yukaichenglab
#SBATCH --account=yukaicheng
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:H800:4
#SBATCH --exclude=gnho009,gnho018
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/code_grid4.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/code_grid4.%j.err
set -Eeuo pipefail
source /storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/code_reasoning_grid_scale8_20260913/common.sh
export GRID_REPLICAS=2
/soft/slurm/bin/srun --nodes=1 --ntasks=1 --cpus-per-task=32 --cpu-bind=cores --kill-on-bad-exit=1 bash "$SCALE_CODE/node.sh"
if [[ ! -f "$GRID_ROOT/ALL_GENERATION_COMPLETE.json" ]]; then
 /soft/slurm/bin/sbatch --parsable --dependency=afterok:$SLURM_JOB_ID "$SCALE_CODE/gpu4.slurm" >> "$GRID_ROOT/job_ids.txt"
fi
