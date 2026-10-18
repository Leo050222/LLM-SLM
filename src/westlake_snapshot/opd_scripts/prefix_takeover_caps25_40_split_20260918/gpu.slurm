#!/bin/bash
#SBATCH -J prefix_caps25_40_split
#SBATCH -p yukaichenglab
#SBATCH --account=yukaicheng
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:H800:4
#SBATCH --exclude=gnho009,gnho018
#SBATCH --mem=384G
#SBATCH --time=UNLIMITED
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/prefix_caps25_40_split.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/prefix_caps25_40_split.%j.err
set -Eeuo pipefail
export PYTHONDONTWRITEBYTECODE=1
export OPD_CODE_ROOT=/storage/yukaichengLab/wangxin/test_eiloeo/opd_online_code/20260911_chunk64_k8_2nodes
export GRID_ROOT=/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_prefix/prefix_takeover_caps25_40_20260918
export SCALE_CODE=/storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/prefix_takeover_caps25_40_split_20260918
export OPD_PYTHON=/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python
export GRID_REPLICAS=2
export EXPECTED_ROWS=800
export OPD_ONLINE_RUN=$GRID_ROOT/slots/job_$SLURM_JOB_ID
[[ ! -f "$GRID_ROOT/ALL_GENERATION_COMPLETE.json" ]] || exit 0
mkdir -p "$OPD_ONLINE_RUN/logs"
nvidia-smi --query-gpu=index,name,memory.used --format=csv
/soft/slurm/bin/srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --cpus-per-task=32 --cpu-bind=cores --kill-on-bad-exit=1 bash "$SCALE_CODE/node.sh"
test -f "$GRID_ROOT/ALL_GENERATION_COMPLETE.json"
if [[ -f "$GRID_ROOT/judge/EXTERNAL_SCORING_APPROVED.json" ]]; then
 /soft/slurm/bin/sbatch --parsable --dependency=afterok:$SLURM_JOB_ID "/storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/prefix_takeover_caps25_40_20260918/score.slurm" >> "$GRID_ROOT/score_job_ids.txt"
else
 /soft/slurm/bin/sbatch --parsable --dependency=afterok:$SLURM_JOB_ID "/storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/prefix_takeover_caps25_40_20260918/code_score.slurm" >> "$GRID_ROOT/score_job_ids.txt"
fi
