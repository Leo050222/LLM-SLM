#!/bin/bash
#SBATCH -J full_baselines_greedy
#SBATCH -p yukaichenglab
#SBATCH --account=yukaicheng
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:H800:8
#SBATCH --exclude=gnho005,gnho008,gnho009,gnho018
#SBATCH --mem=512G
#SBATCH --time=01:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/full_baselines_greedy.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/full_baselines_greedy.%j.err
set -Eeuo pipefail
export PYTHONDONTWRITEBYTECODE=1
export OPD_CODE_ROOT=/storage/yukaichengLab/wangxin/test_eiloeo/opd_online_code/20260911_chunk64_k8_2nodes
export GRID_ROOT=/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_baselines/full_teacher_student_greedy_20260920
export SCALE_CODE=/storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/full_teacher_student_greedy_8gpu_20260921
export OPD_PYTHON=/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python
export GRID_REPLICAS=8
export OPD_ONLINE_RUN=$GRID_ROOT/slots/job_${SLURM_JOB_ID}_preflight
mkdir -p "$OPD_ONLINE_RUN/logs"
PREFLIGHT_ONLY=1 /soft/slurm/bin/srun --nodes=1 --ntasks=1 --cpus-per-task=64 --cpu-bind=cores --kill-on-bad-exit=1 bash "$SCALE_CODE/node.sh"
"$OPD_PYTHON" "$SCALE_CODE/cutover.py"
for BASELINE_ROLE in student teacher; do
 export BASELINE_ROLE
 [[ ! -f "$GRID_ROOT/${BASELINE_ROLE}_GENERATION_COMPLETE.json" ]] || continue
 export OPD_ONLINE_RUN=$GRID_ROOT/slots/job_${SLURM_JOB_ID}_${BASELINE_ROLE}
 mkdir -p "$OPD_ONLINE_RUN/logs"
 /soft/slurm/bin/srun --nodes=1 --ntasks=1 --cpus-per-task=64 --cpu-bind=cores --kill-on-bad-exit=1 bash "$SCALE_CODE/node.sh"
 test -f "$GRID_ROOT/${BASELINE_ROLE}_GENERATION_COMPLETE.json"
done
test -f "$GRID_ROOT/ALL_GENERATION_COMPLETE.json"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$OPD_PYTHON" /storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/full_teacher_student_greedy_20260920/score.py --code-only
