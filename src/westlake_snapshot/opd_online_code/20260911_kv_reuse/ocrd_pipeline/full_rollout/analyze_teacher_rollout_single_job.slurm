#!/usr/bin/env bash
#SBATCH --job-name=score_tfull_single
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/dataset/train/teacher_full_qwen3_14b_n8_seed20260826/logs/%j.analysis_single.out
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/dataset/train/teacher_full_qwen3_14b_n8_seed20260826/logs/%j.analysis_single.err

set -euo pipefail
ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
ROLLOUT=${ROOT}/dataset/train/teacher_full_qwen3_14b_n8_seed20260826
DATA=${ROOT}/dataset/train/original_deepmath_ge6_train25pct/train_unique.parquet
OUTPUT=${ROLLOUT}/analysis_correctness_length_single_job
PYTHON=/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python
NUM_SHARDS=64
LOCAL_PARALLELISM=16
RUN_ROOT=${OUTPUT}/runs/${SLURM_JOB_ID}

test -f "${ROLLOUT}/FULL_ROLLOUT_BATCH_COMPLETE"
test ! -e "${OUTPUT}/TEACHER_ROLLOUT_ANALYSIS_COMPLETE"
mkdir -p "${OUTPUT}/logs" "${RUN_ROOT}/logs" "${RUN_ROOT}/status"
trap 'rc=$?; echo "TEACHER_ANALYSIS_SLURM_ERROR job=${SLURM_JOB_ID:-unset} line=${LINENO} command=${BASH_COMMAND} exit_code=${rc}" >&2' ERR

score_shard() {
  local shard_index=$1
  local started_at finished_at rc mode temporary marker
  started_at=$(date --iso-8601=seconds)
  marker=${OUTPUT}/shards/$(printf 'shard_%03d' "${shard_index}")/SCORE_SHARD_COMPLETE
  temporary=${RUN_ROOT}/status/shard_${shard_index}.exit.tmp.$$
  if [[ -f "${marker}" ]]; then
    rc=0
    mode=cached
    echo "ANALYSIS_SHARD_REUSED shard=${shard_index} marker=${marker}"
  else
    mode=scored
    set +e
    "${PYTHON}" "${ROOT}/ocrd_pipeline/full_rollout/analyze_teacher_rollout_shard.py" \
      --rollout-root "${ROLLOUT}" \
      --dataset "${DATA}" \
      --output-root "${OUTPUT}" \
      --shard-index "${shard_index}" \
      --num-shards "${NUM_SHARDS}" \
      --num-rollouts 8 \
      --score-timeout 20
    rc=$?
    set -e
  fi
  finished_at=$(date --iso-8601=seconds)
  printf 'shard\t%s\nexit_code\t%s\nmode\t%s\nstarted_at\t%s\nfinished_at\t%s\n' \
    "${shard_index}" "${rc}" "${mode}" "${started_at}" "${finished_at}" >"${temporary}"
  mv "${temporary}" "${RUN_ROOT}/status/shard_${shard_index}.exit"
  return "${rc}"
}

run_wave() {
  local first=$1
  local last=$2
  local shard_index rc
  local -a pids=()
  local -a shards=()
  for shard_index in $(seq "${first}" "${last}"); do
    score_shard "${shard_index}" \
      >"${RUN_ROOT}/logs/shard_${shard_index}.out" \
      2>"${RUN_ROOT}/logs/shard_${shard_index}.err" &
    pids+=("$!")
    shards+=("${shard_index}")
  done
  set +e
  for offset in "${!pids[@]}"; do
    wait "${pids[$offset]}"
    rc=$?
    shard_index=${shards[$offset]}
    printf '%s\t%s\t%s\t%s\n' "${shard_index}" "${pids[$offset]}" "${rc}" \
      "${RUN_ROOT}/status/shard_${shard_index}.exit" >>"${RUN_ROOT}/worker_exit_status.tsv"
    if [[ "${rc}" -ne 0 ]]; then
      printf '%s\t%s\t%s\n' "${shard_index}" "${rc}" \
        "${RUN_ROOT}/logs/shard_${shard_index}.err" >>"${RUN_ROOT}/failed_shards.tsv"
    fi
  done
  set -e
}

printf 'shard\tpid\texit_code\tstatus_file\n' >"${RUN_ROOT}/worker_exit_status.tsv"
printf 'shard\texit_code\tstderr_file\n' >"${RUN_ROOT}/failed_shards.tsv"
for first in $(seq 0 "${LOCAL_PARALLELISM}" $((NUM_SHARDS - 1))); do
  last=$((first + LOCAL_PARALLELISM - 1))
  if [[ "${last}" -ge "${NUM_SHARDS}" ]]; then
    last=$((NUM_SHARDS - 1))
  fi
  echo "ANALYSIS_WAVE_START job=${SLURM_JOB_ID} first=${first} last=${last} parallelism=${LOCAL_PARALLELISM}"
  run_wave "${first}" "${last}"
  echo "ANALYSIS_WAVE_END job=${SLURM_JOB_ID} first=${first} last=${last} completed_markers=$(find "${OUTPUT}/shards" -type f -name SCORE_SHARD_COMPLETE 2>/dev/null | wc -l)"
done

failed_count=$(( $(wc -l <"${RUN_ROOT}/failed_shards.tsv") - 1 ))
if [[ "${failed_count}" -ne 0 ]]; then
  echo "TEACHER_ANALYSIS_SHARDS_FAILED job=${SLURM_JOB_ID} failed_count=${failed_count} failure_table=${RUN_ROOT}/failed_shards.tsv" >&2
  exit 20
fi

marker_count=$(find "${OUTPUT}/shards" -type f -name SCORE_SHARD_COMPLETE | wc -l)
if [[ "${marker_count}" -ne "${NUM_SHARDS}" ]]; then
  echo "TEACHER_ANALYSIS_MARKER_COUNT_MISMATCH expected=${NUM_SHARDS} actual=${marker_count}" >&2
  exit 21
fi
"${PYTHON}" "${ROOT}/ocrd_pipeline/full_rollout/merge_teacher_rollout_analysis.py" \
  --rollout-root "${ROLLOUT}" \
  --analysis-root "${OUTPUT}" \
  --num-shards "${NUM_SHARDS}" \
  --expected-traces 13244 \
  --num-rollouts 8
test -f "${OUTPUT}/TEACHER_ROLLOUT_ANALYSIS_COMPLETE"
echo "TEACHER_ROLLOUT_SINGLE_CPU_JOB_COMPLETE job=${SLURM_JOB_ID}"
