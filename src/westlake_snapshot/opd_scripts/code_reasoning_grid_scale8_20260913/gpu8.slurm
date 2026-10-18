#!/bin/bash
#SBATCH -J code_grid8
#SBATCH -p yukaichenglab
#SBATCH --account=yukaicheng
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:H800:8
#SBATCH --nodelist=gnho020
#SBATCH --mem=512G
#SBATCH --time=06:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/code_grid8.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/code_grid8.%j.err
set -Eeuo pipefail
source /storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/code_reasoning_grid_scale8_20260913/common.sh
export GRID_REPLICAS=4
cutoff=$(TZ=Asia/Shanghai date -d '2026-09-13 10:00:00' +%s)
now=$(date +%s)
if (( now >= cutoff )); then exit 0; fi
# Absolute wall-clock cap; the dependent4-GPU job already exists before launch.
(sleep "$((cutoff-now))"; /soft/slurm/bin/scancel "$SLURM_JOB_ID") &
timer=$!
trap 'kill "$timer" 2>/dev/null || true' EXIT
/soft/slurm/bin/srun --nodes=1 --ntasks=1 --cpus-per-task=64 --cpu-bind=cores --kill-on-bad-exit=1 bash "$SCALE_CODE/node.sh"
