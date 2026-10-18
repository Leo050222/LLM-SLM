#!/bin/bash
set -Eeuo pipefail
S=/storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/full_teacher_student_greedy_20260920
P=/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_baselines/full_teacher_student_greedy_20260920
PY=/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python
mkdir -p "$P" /storage/yukaichengLab/wangxin/test_eiloeo/opd_logs
exec 9>"$P/submit.lock"
flock -n 9
test ! -f "$P/serial_job_id.txt"
"$PY" "$S/prepare.py"
cd "$S"
"$PY" -m unittest test_generation test_greedy
job=$(/soft/slurm/bin/sbatch --parsable --job-name=greedy_baselines_serial "$S/gpu.slurm")
printf '%s\n' "$job" > "$P/serial_job_id.txt"
/soft/slurm/bin/squeue -j "$job" -o '%.12i %.24j %.8T %.25N %.30R'
