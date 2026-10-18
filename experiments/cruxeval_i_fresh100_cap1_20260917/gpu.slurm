#!/bin/bash
#SBATCH -J crux_i_fresh100_cap1
#SBATCH -p yukaichenglab
#SBATCH --account=yukaicheng
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:H800:4
#SBATCH --exclude=gnho009,gnho018
#SBATCH --mem=256G
#SBATCH --time=UNLIMITED
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/crux_i_fresh100_cap1.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/crux_i_fresh100_cap1.%j.err
set -Eeuo pipefail
export PYTHONDONTWRITEBYTECODE=1
export OPD_CODE_ROOT=/storage/yukaichengLab/wangxin/test_eiloeo/opd_online_code/20260911_chunk64_k8_2nodes
export GRID_ROOT=/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_prefix/cruxeval_i_fresh100_cap1_20260917
export SCALE_CODE=/storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/cruxeval_i_fresh100_cap1_20260917
export OPD_PYTHON=/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python
export GRID_REPLICAS=2
export EXPECTED_ROWS=100
export OPD_ONLINE_RUN=$GRID_ROOT/slots/job_$SLURM_JOB_ID
[[ ! -f "$GRID_ROOT/ALL_GENERATION_COMPLETE.json" ]] || exit 0
mkdir -p "$OPD_ONLINE_RUN/logs"
nvidia-smi --query-gpu=index,name,memory.used --format=csv
/soft/slurm/bin/srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --cpus-per-task=32 --cpu-bind=cores --kill-on-bad-exit=1 bash "$SCALE_CODE/node.sh"
test -f "$GRID_ROOT/ALL_GENERATION_COMPLETE.json"
/soft/slurm/bin/sbatch --parsable --dependency=afterok:$SLURM_JOB_ID "$SCALE_CODE/score.slurm" >> "$GRID_ROOT/score_job_ids.txt"
