#!/bin/bash
#SBATCH -J quartile_greedy_baselines
#SBATCH -p yukaichenglab
#SBATCH --account=yukaicheng
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:H800:8
#SBATCH --exclude=gnho009,gnho018
#SBATCH --mem=512G
#SBATCH --time=UNLIMITED
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/quartile_greedy_baselines.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/quartile_greedy_baselines.%j.err
set -Eeuo pipefail
export PYTHONDONTWRITEBYTECODE=1
export OPD_CODE_ROOT=/storage/yukaichengLab/wangxin/test_eiloeo/opd_online_code/20260911_chunk64_k8_2nodes
export GRID_ROOT=/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_prefix/quartile_support_k_20260918
export SCALE_CODE=/storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/quartile_greedy_baselines_20260918
export OPD_PYTHON=/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python
export GRID_REPLICAS=8

export EXPECTED_ROWS=800
for BASELINE_ROLE in student teacher; do
 export BASELINE_ROLE
 if [[ "$BASELINE_ROLE" == student ]]; then group=k0; else group=teacher_greedy; fi
 [[ ! -f "$GRID_ROOT/${group}_GENERATION_COMPLETE.json" ]] || continue
 export OPD_ONLINE_RUN=$GRID_ROOT/slots/job_${SLURM_JOB_ID}_${group}
 mkdir -p "$OPD_ONLINE_RUN/logs"
 /soft/slurm/bin/srun --nodes=1 --ntasks=1 --cpus-per-task=64 --cpu-bind=cores --kill-on-bad-exit=1 bash "$SCALE_CODE/node.sh"
 test -f "$GRID_ROOT/${group}_GENERATION_COMPLETE.json"
done
