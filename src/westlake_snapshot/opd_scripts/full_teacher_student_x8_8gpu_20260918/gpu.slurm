#!/bin/bash
#SBATCH -J full_baselines_x8_8gpu
#SBATCH -p yukaichenglab
#SBATCH --account=yukaicheng
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:H800:8
#SBATCH --nodelist=gnho017
#SBATCH --mem=512G
#SBATCH --time=UNLIMITED
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/full_baselines_x8_8gpu.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/full_baselines_x8_8gpu.%j.err
set -Eeuo pipefail
export PYTHONDONTWRITEBYTECODE=1
export OPD_CODE_ROOT=/storage/yukaichengLab/wangxin/test_eiloeo/opd_online_code/20260911_chunk64_k8_2nodes
export GRID_ROOT=/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_baselines/full_teacher_student_x8_20260918
export SCALE_CODE=/storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/full_teacher_student_x8_8gpu_20260918
export OPD_PYTHON=/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python
export GRID_REPLICAS=8
[[ ! -f "$GRID_ROOT/ALL_GENERATION_COMPLETE.json" ]] || exit 0
for BASELINE_ROLE in student teacher; do
 export BASELINE_ROLE
 [[ ! -f "$GRID_ROOT/${BASELINE_ROLE}_GENERATION_COMPLETE.json" ]] || continue
 export OPD_ONLINE_RUN=$GRID_ROOT/slots/job_${SLURM_JOB_ID}_${BASELINE_ROLE}
 mkdir -p "$OPD_ONLINE_RUN/logs"
 /soft/slurm/bin/srun --nodes=1 --ntasks=1 --cpus-per-task=64 --cpu-bind=cores --kill-on-bad-exit=1 bash "$SCALE_CODE/node.sh"
 test -f "$GRID_ROOT/${BASELINE_ROLE}_GENERATION_COMPLETE.json"
done
test -f "$GRID_ROOT/ALL_GENERATION_COMPLETE.json"
if [[ -f "$GRID_ROOT/judge/EXTERNAL_SCORING_APPROVED.json" ]]; then
 /soft/slurm/bin/sbatch --parsable --dependency=afterok:$SLURM_JOB_ID "/storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/full_teacher_student_x8_20260918/score.slurm" >> "$GRID_ROOT/score_job_ids.txt"
else
 /soft/slurm/bin/sbatch --parsable --dependency=afterok:$SLURM_JOB_ID "/storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/full_teacher_student_x8_20260918/code_score.slurm" >> "$GRID_ROOT/score_job_ids.txt"
fi
