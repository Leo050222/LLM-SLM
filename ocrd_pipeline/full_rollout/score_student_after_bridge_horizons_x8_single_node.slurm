#!/usr/bin/env bash
#SBATCH --job-name=sbridge_horz_score_1n
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=768G
#SBATCH --time=1-12:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/sbridge_horz_score_1n.%j.out
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/sbridge_horz_score_1n.%j.err

set -euo pipefail

ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
PYTHON=/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python
MODULE=${ROOT}/ocrd_pipeline/full_rollout
RUN_ROOT=${ROOT}/opd_logs/sbridge_horz_score_1n.${SLURM_JOB_ID}
HORIZONS=(32 64 128)
TASK_PARALLELISM=4
SHARD_PARALLELISM=32

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/status"
trap 'rc=$?; echo "SBRIDGE_HORIZON_SCORE_1N_ERROR job=${SLURM_JOB_ID:-unset} line=${LINENO} command=${BASH_COMMAND} exit_code=${rc}" >&2' ERR

echo "SBRIDGE_HORIZON_SCORE_1N_START job=${SLURM_JOB_ID} node=${SLURMD_NODENAME} cpus=${SLURM_CPUS_ON_NODE:-unset} time=$(date --iso-8601=seconds)"

score_task() {
  local task_id=$1 horizon_index replicate h parent rollout marker temporary rc mode
  horizon_index=$((task_id / 8))
  replicate=$((task_id % 8))
  h=${HORIZONS[${horizon_index}]}
  parent=${ROOT}/dataset/train/student_after_bridge_qwen3_1_7b_h${h}_n8_x8_seed20260826
  rollout=${parent}/$(printf 'replicate_%02d' "${replicate}")
  marker=${rollout}/analysis_correctness_length_single_job/STUDENT_AFTER_BRIDGE_ROLLOUT_ANALYSIS_COMPLETE
  temporary=${RUN_ROOT}/status/task_${task_id}.tmp.$$
  if [[ -f "${marker}" ]]; then
    mode=cached
    rc=0
  else
    mode=resume_or_score
    set +e
    LOCAL_PARALLELISM=${SHARD_PARALLELISM} bash "${MODULE}/score_student_after_bridge_horizon_root.sh" \
      "${rollout}" "$((20260826 + replicate * 1000000))" "${h}"
    rc=$?
    set -e
  fi
  printf 'task\t%s\nhorizon\t%s\nreplicate\t%s\nmode\t%s\nexit_code\t%s\n' \
    "${task_id}" "${h}" "${replicate}" "${mode}" "${rc}" >"${temporary}"
  mv "${temporary}" "${RUN_ROOT}/status/task_${task_id}.tsv"
  return "${rc}"
}

run_wave() {
  local first=$1 last=$2 offset task_id rc failed
  local -a pids=() tasks=()
  failed=0
  for task_id in $(seq "${first}" "${last}"); do
    score_task "${task_id}" >"${RUN_ROOT}/logs/task_${task_id}.out" \
      2>"${RUN_ROOT}/logs/task_${task_id}.err" &
    pids+=("$!")
    tasks+=("${task_id}")
  done
  set +e
  for offset in "${!pids[@]}"; do
    wait "${pids[$offset]}"
    rc=$?
    task_id=${tasks[$offset]}
    printf '%s\t%s\t%s\n' "${task_id}" "${pids[$offset]}" "${rc}" >>"${RUN_ROOT}/task_exit_status.tsv"
    [[ "${rc}" -eq 0 ]] || failed=1
  done
  set -e
  [[ "${failed}" -eq 0 ]]
}

printf 'task\tpid\texit_code\n' >"${RUN_ROOT}/task_exit_status.tsv"
for first in $(seq 0 "${TASK_PARALLELISM}" 23); do
  last=$((first + TASK_PARALLELISM - 1))
  [[ "${last}" -le 23 ]] || last=23
  echo "SBRIDGE_HORIZON_SCORE_1N_WAVE_START first=${first} last=${last} time=$(date --iso-8601=seconds)"
  run_wave "${first}" "${last}"
  echo "SBRIDGE_HORIZON_SCORE_1N_WAVE_COMPLETE first=${first} last=${last} time=$(date --iso-8601=seconds)"
done

for h in "${HORIZONS[@]}"; do
  parent=${ROOT}/dataset/train/student_after_bridge_qwen3_1_7b_h${h}_n8_x8_seed20260826
  for replicate in $(seq 0 7); do
    test -f "${parent}/$(printf 'replicate_%02d' "${replicate}")/analysis_correctness_length_single_job/STUDENT_AFTER_BRIDGE_ROLLOUT_ANALYSIS_COMPLETE"
  done
  if [[ ! -f "${parent}/aggregate/STUDENT_AFTER_BRIDGE_X8_AGGREGATE_COMPLETE" ]]; then
    "${PYTHON}" "${MODULE}/aggregate_student_after_bridge_x8.py" \
      --original-root "${parent}/replicate_00" \
      --replicate-parent "${parent}" \
      --output-root "${parent}/aggregate" \
      --bridge-token-limit "${h}"
  fi
  test -f "${parent}/aggregate/STUDENT_AFTER_BRIDGE_X8_AGGREGATE_COMPLETE"
done

touch "${RUN_ROOT}/SBRIDGE_HORIZON_SCORE_1N_COMPLETE"
echo "SBRIDGE_HORIZON_SCORE_1N_COMPLETE job=${SLURM_JOB_ID} node=${SLURMD_NODENAME} time=$(date --iso-8601=seconds)"
