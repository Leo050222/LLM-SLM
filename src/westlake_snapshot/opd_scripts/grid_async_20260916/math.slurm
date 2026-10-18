#!/bin/bash
#SBATCH -J math_grid_async
#SBATCH -p yukaichenglab
#SBATCH --account=yukaicheng
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:H800:8
#SBATCH --exclusive
#SBATCH --exclude=gnho009,gnho018
#SBATCH --mem=512G
#SBATCH --time=06:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/math_grid_async.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/math_grid_async.%j.err
set -Eeuo pipefail
export PYTHONDONTWRITEBYTECODE=1
export OPD_CODE_ROOT=/storage/yukaichengLab/wangxin/test_eiloeo/opd_online_code/20260911_chunk64_k8_2nodes
export GRID_ROOT=/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_hard/math_reasoning_grid_x8_20260916
export SCALE_CODE=/storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/grid_async_20260916
export OPD_PYTHON=/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python
export GRID_REPLICAS=4
export EXPECTED_ROWS=2254
export OPD_ONLINE_RUN=$GRID_ROOT/slots/job_$SLURM_JOB_ID
[[ ! -f "$GRID_ROOT/ALL_GENERATION_COMPLETE.json" ]] || exit 0
mkdir -p "$OPD_ONLINE_RUN/logs"
nvidia-smi --query-gpu=index,name,memory.used --format=csv
# Benchmark occupies only this existing eight-GPU allocation.
if [[ "${RUN_ASYNC_BENCHMARK:-0}" == 1 ]]; then
 export OPD_ONLINE_RUN=$GRID_ROOT/slots/job_${SLURM_JOB_ID}_bench
 mkdir -p "$OPD_ONLINE_RUN/logs"
 export BENCHMARK_MODE=1
 if /soft/slurm/bin/srun --nodes=1 --ntasks=1 --cpus-per-task=64 --cpu-bind=cores --kill-on-bad-exit=1 bash "$SCALE_CODE/bench_node.sh"; then
  cp "$OPD_ONLINE_RUN/benchmark_summary.json" "$GRID_ROOT/recovery_async_v3/benchmark_summary.json"
 else
  test ! -f "$OPD_ONLINE_RUN/benchmark_summary.json" || cp "$OPD_ONLINE_RUN/benchmark_summary.json" "$GRID_ROOT/recovery_async_v3/benchmark_summary.json"
  export SCALE_CODE=/storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/grid_fifo_20260916
 fi
 unset BENCHMARK_MODE RUN_ASYNC_BENCHMARK
fi
export OPD_ONLINE_RUN=$GRID_ROOT/slots/job_$SLURM_JOB_ID
mkdir -p "$OPD_ONLINE_RUN/logs"
/soft/slurm/bin/srun --nodes=1 --ntasks=1 --cpus-per-task=64 --cpu-bind=cores --kill-on-bad-exit=1 bash "$SCALE_CODE/node.sh"
if [[ ! -f "$GRID_ROOT/ALL_GENERATION_COMPLETE.json" ]]; then
 /soft/slurm/bin/sbatch --parsable --dependency=afterok:$SLURM_JOB_ID "$SCALE_CODE/math.slurm" >> "$GRID_ROOT/job_ids.txt"
fi
