#!/bin/bash
set -euo pipefail
export OPD_CODE_ROOT=/storage/yukaichengLab/wangxin/test_eiloeo/opd_online_code/20260911_random_k8_cap8
export OPD_ONLINE_RUN=/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_deepmath/random_k8_cap8_test1000_h100_seed20260911
export OPD_PYTHON=/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python
test ! -f "$OPD_ONLINE_RUN/gpu_job.id"
"$OPD_PYTHON" "$OPD_ONLINE_RUN/prepare_random.py"
gpu_job=$(/soft/slurm/bin/sbatch --parsable --output="$OPD_ONLINE_RUN/logs/%j.out" --error="$OPD_ONLINE_RUN/logs/%j.err" "$OPD_ONLINE_RUN/workers.slurm")
echo "$gpu_job" > "$OPD_ONLINE_RUN/gpu_job.id"
nohup bash "$OPD_ONLINE_RUN/coordinator.sh" > "$OPD_ONLINE_RUN/logs/coordinator.log" 2>&1 < /dev/null &
echo $! > "$OPD_ONLINE_RUN/coordinator.pid"
score_job=$(/soft/slurm/bin/sbatch --parsable --dependency="afterok:$gpu_job" --output="$OPD_ONLINE_RUN/logs/%A_%a_score.out" --error="$OPD_ONLINE_RUN/logs/%A_%a_score.err" "$OPD_ONLINE_RUN/score_sharded.slurm")
echo "$score_job" > "$OPD_ONLINE_RUN/score_job.id"
aggregate=$(/soft/slurm/bin/sbatch --parsable --dependency="afterok:$score_job" --cpus-per-task=4 --mem=16G --output="$OPD_ONLINE_RUN/logs/%j_aggregate.out" --error="$OPD_ONLINE_RUN/logs/%j_aggregate.err" "$OPD_ONLINE_RUN/aggregate.slurm")
echo "$aggregate" > "$OPD_ONLINE_RUN/aggregate_job.id"
echo "GPU_JOB=$gpu_job SCORE_ARRAY=$score_job AGGREGATE=$aggregate"
