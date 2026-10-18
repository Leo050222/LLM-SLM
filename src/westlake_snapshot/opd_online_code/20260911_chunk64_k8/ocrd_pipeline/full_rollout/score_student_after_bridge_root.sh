#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 ROLLOUT_ROOT BASE_SEED" >&2
  exit 2
fi

ROLLOUT=$1
BASE_SEED=$2
ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
DATA=${ROOT}/dataset/train/original_deepmath_ge6_train25pct/train_unique.parquet
OUTPUT=${ROLLOUT}/analysis_correctness_length_single_job
PYTHON=/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python
NUM_SHARDS=64
LOCAL_PARALLELISM=16
RUN_ROOT=${OUTPUT}/runs/${SLURM_JOB_ID}
FINAL_MARKER=STUDENT_AFTER_BRIDGE_ROLLOUT_ANALYSIS_COMPLETE

test -f "${ROLLOUT}/GPU_GENERATION_COMPLETE"
test "$(find "${ROLLOUT}/shards" -type f -name SHARD_COMPLETE | wc -l)" -eq 8
if [[ ! -f "${ROLLOUT}/FULL_ROLLOUT_BATCH_COMPLETE" ]]; then
  "${PYTHON}" "${ROOT}/ocrd_pipeline/full_rollout/verify_full_rollout_batch.py" \
    --mode student_after_bridge \
    --bridge-manifest "${ROOT}/dataset/train/anchor_bridge_qwen3_14b_h256_n8/manifest.jsonl" \
    --output-root "${ROLLOUT}" \
    --num-shards 8 \
    --num-rollouts 8 \
    --model /storage/yukaichengLab/share/model/Qwen3-1.7B \
    --max-model-len 40960 \
    --seed "${BASE_SEED}" \
    --expected-count 13244
fi
test -f "${ROLLOUT}/FULL_ROLLOUT_BATCH_COMPLETE"
[[ ! -e "${OUTPUT}/${FINAL_MARKER}" ]] || { echo "Refusing to overwrite completed analysis ${OUTPUT}" >&2; exit 7; }
mkdir -p "${OUTPUT}/logs" "${RUN_ROOT}/logs" "${RUN_ROOT}/status"
trap 'rc=$?; echo "STUDENT_BRIDGE_REPLICATE_ANALYSIS_ERROR job=${SLURM_JOB_ID:-unset} line=${LINENO} command=${BASH_COMMAND} exit_code=${rc}" >&2' ERR

score_shard() {
  local shard_index=$1 started_at finished_at rc mode temporary marker
  started_at=$(date --iso-8601=seconds)
  marker=${OUTPUT}/shards/$(printf 'shard_%03d' "${shard_index}")/SCORE_SHARD_COMPLETE
  temporary=${RUN_ROOT}/status/shard_${shard_index}.exit.tmp.$$
  if [[ -f "${marker}" ]]; then
    rc=0
    mode=cached
  else
    mode=scored
    set +e
    "${PYTHON}" "${ROOT}/ocrd_pipeline/full_rollout/analyze_teacher_rollout_shard.py" \
      --rollout-kind student_after_bridge \
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
  local first=$1 last=$2 shard_index rc offset
  local -a pids=() shards=()
  for shard_index in $(seq "${first}" "${last}"); do
    score_shard "${shard_index}" >"${RUN_ROOT}/logs/shard_${shard_index}.out" 2>"${RUN_ROOT}/logs/shard_${shard_index}.err" &
    pids+=("$!")
    shards+=("${shard_index}")
  done
  set +e
  for offset in "${!pids[@]}"; do
    wait "${pids[$offset]}"; rc=$?
    shard_index=${shards[$offset]}
    printf '%s\t%s\t%s\t%s\n' "${shard_index}" "${pids[$offset]}" "${rc}" "${RUN_ROOT}/status/shard_${shard_index}.exit" >>"${RUN_ROOT}/worker_exit_status.tsv"
    [[ "${rc}" -eq 0 ]] || printf '%s\t%s\t%s\n' "${shard_index}" "${rc}" "${RUN_ROOT}/logs/shard_${shard_index}.err" >>"${RUN_ROOT}/failed_shards.tsv"
  done
  set -e
}

printf 'shard\tpid\texit_code\tstatus_file\n' >"${RUN_ROOT}/worker_exit_status.tsv"
printf 'shard\texit_code\tstderr_file\n' >"${RUN_ROOT}/failed_shards.tsv"
for first in $(seq 0 "${LOCAL_PARALLELISM}" $((NUM_SHARDS - 1))); do
  last=$((first + LOCAL_PARALLELISM - 1))
  [[ "${last}" -lt "${NUM_SHARDS}" ]] || last=$((NUM_SHARDS - 1))
  run_wave "${first}" "${last}"
done

failed_count=$(( $(wc -l <"${RUN_ROOT}/failed_shards.tsv") - 1 ))
[[ "${failed_count}" -eq 0 ]] || { echo "Analysis shards failed: ${failed_count}" >&2; exit 20; }
marker_count=$(find "${OUTPUT}/shards" -type f -name SCORE_SHARD_COMPLETE | wc -l)
[[ "${marker_count}" -eq "${NUM_SHARDS}" ]] || { echo "Score marker mismatch: ${marker_count}/${NUM_SHARDS}" >&2; exit 21; }
"${PYTHON}" "${ROOT}/ocrd_pipeline/full_rollout/merge_teacher_rollout_analysis.py" \
  --rollout-kind student_after_bridge \
  --rollout-root "${ROLLOUT}" \
  --analysis-root "${OUTPUT}" \
  --num-shards "${NUM_SHARDS}" \
  --expected-traces 13244 \
  --num-rollouts 8
test -f "${OUTPUT}/${FINAL_MARKER}"
echo "STUDENT_BRIDGE_REPLICATE_ANALYSIS_COMPLETE job=${SLURM_JOB_ID} root=${ROLLOUT}"
