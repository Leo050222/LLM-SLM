#!/bin/bash
#SBATCH -J random_quartile_greedy
#SBATCH -p yukaichenglab
#SBATCH --account=yukaicheng
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:H800:6
#SBATCH --nodelist=gnho031
#SBATCH --mem=384G
#SBATCH --time=02:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/random_quartile_greedy.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/random_quartile_greedy.%j.err
set -Eeuo pipefail
export PYTHONDONTWRITEBYTECODE=1
export OPD_CODE_ROOT=/storage/yukaichengLab/wangxin/test_eiloeo/opd_online_code/20260911_chunk64_k8_2nodes
export GRID_ROOT=/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_ablation/random_quartile_greedy_k1_20260921
export SCALE_CODE=/storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/random_quartile_greedy_k1_20260921
export OPD_PYTHON=/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python
export GRID_REPLICAS=3
export EXPECTED_ROWS=4571
export OPD_ONLINE_RUN=$GRID_ROOT/slots/job_$SLURM_JOB_ID
[[ ! -f "$GRID_ROOT/ALL_GENERATION_COMPLETE.json" ]] || exit 0
mkdir -p "$OPD_ONLINE_RUN/logs"
nvidia-smi --query-gpu=index,name,memory.used --format=csv
/soft/slurm/bin/srun --nodes=1 --ntasks=1 --cpus-per-task=48 --cpu-bind=cores --kill-on-bad-exit=1 bash "$SCALE_CODE/node.sh"
test -f "$GRID_ROOT/ALL_GENERATION_COMPLETE.json"
