#!/usr/bin/env bash
#SBATCH --job-name=sbridge_horz_score_3n
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=1-12:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/sbridge_horz_score_3n.%j.out
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/sbridge_horz_score_3n.%j.err

set -euo pipefail

: "${WORKER_INDEX:?WORKER_INDEX must be set to 0, 1, or 2}"
: "${RUN_GROUP:?RUN_GROUP must be set}"
[[ "${WORKER_INDEX}" =~ ^[0-2]$ ]]
[[ "${RUN_GROUP}" =~ ^[A-Za-z0-9_.-]+$ ]]

ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
PYTHON=/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python
MODULE=${ROOT}/ocrd_pipeline/full_rollout
RUN_ROOT=${ROOT}/opd_logs/${RUN_GROUP}
HORIZONS=(32 64 128)
TOTAL_WORKERS=3
SHARD_PARALLELISM=64

# Each shard scorer is a separate process.  Keep numerical libraries from
# creating nested thread pools beyond the 64 CPUs granted by Slurm.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/status"
trap 'rc=$?; echo "SBRIDGE_HORIZON_SCORE_3N_ERROR job=${SLURM_JOB_ID:-unset} worker=${WORKER_INDEX:-unset} line=${LINENO} command=${BASH_COMMAND} exit_code=${rc}" >&2' ERR

echo "SBRIDGE_HORIZON_SCORE_3N_START job=${SLURM_JOB_ID} worker=${WORKER_INDEX} node=${SLURMD_NODENAME} cpus=${SLURM_CPUS_ON_NODE:-unset} time=$(date --iso-8601=seconds)"

QUEUE=${RUN_ROOT}/worker_${WORKER_INDEX}_pending_shards.tsv
: >"${QUEUE}"
for ((task_id=WORKER_INDEX; task_id<24; task_id+=TOTAL_WORKERS)); do
  horizon_index=$((task_id / 8))
  replicate=$((task_id % 8))
  h=${HORIZONS[${horizon_index}]}
  parent=${ROOT}/dataset/train/student_after_bridge_qwen3_1_7b_h${h}_n8_x8_seed20260826
  rollout=${parent}/$(printf 'replicate_%02d' "${replicate}")
  output=${rollout}/analysis_correctness_length_single_job
  final_marker=${output}/STUDENT_AFTER_BRIDGE_ROLLOUT_ANALYSIS_COMPLETE
  [[ -f "${final_marker}" ]] && continue

  test -f "${rollout}/GPU_GENERATION_COMPLETE"
  test "$(find "${rollout}/shards" -type f -name SHARD_COMPLETE | wc -l)" -eq 8
  if [[ ! -f "${rollout}/FULL_ROLLOUT_BATCH_COMPLETE" ]]; then
    "${PYTHON}" "${MODULE}/verify_full_rollout_batch.py" \
      --mode student_after_bridge \
      --bridge-manifest "${ROOT}/dataset/train/anchor_bridge_qwen3_14b_h256_n8/manifest.jsonl" \
      --output-root "${rollout}" --num-shards 8 --num-rollouts 8 \
      --bridge-token-limit "${h}" \
      --model /storage/yukaichengLab/share/model/Qwen3-1.7B \
      --max-model-len 40960 --seed "$((20260826 + replicate * 1000000))" \
      --expected-count 13244
  fi
  test -f "${rollout}/FULL_ROLLOUT_BATCH_COMPLETE"
  for shard_index in $(seq 0 63); do
    shard_marker=${output}/shards/$(printf 'shard_%03d' "${shard_index}")/SCORE_SHARD_COMPLETE
    [[ -f "${shard_marker}" ]] || printf '%s\t%s\t%s\t%s\n' \
      "${h}" "${replicate}" "${shard_index}" "${RUN_ROOT}" >>"${QUEUE}"
  done
done

pending_shards=$(wc -l <"${QUEUE}")
echo "SBRIDGE_HORIZON_SCORE_3N_QUEUE worker=${WORKER_INDEX} pending_shards=${pending_shards} parallelism=${SHARD_PARALLELISM}"
if [[ "${pending_shards}" -gt 0 ]]; then
  xargs -P "${SHARD_PARALLELISM}" -n 4 \
    bash "${MODULE}/score_student_after_bridge_horizon_shard.sh" <"${QUEUE}"
fi

# Merge each replicate only after every assigned missing shard has completed.
for ((task_id=WORKER_INDEX; task_id<24; task_id+=TOTAL_WORKERS)); do
  horizon_index=$((task_id / 8))
  replicate=$((task_id % 8))
  h=${HORIZONS[${horizon_index}]}
  parent=${ROOT}/dataset/train/student_after_bridge_qwen3_1_7b_h${h}_n8_x8_seed20260826
  rollout=${parent}/$(printf 'replicate_%02d' "${replicate}")
  output=${rollout}/analysis_correctness_length_single_job
  final_marker=${output}/STUDENT_AFTER_BRIDGE_ROLLOUT_ANALYSIS_COMPLETE
  mode=cached_complete
  if [[ ! -f "${final_marker}" ]]; then
    mode=merged_after_resume_or_score
    test "$(find "${output}/shards" -type f -name SCORE_SHARD_COMPLETE | wc -l)" -eq 64
    "${PYTHON}" "${MODULE}/merge_teacher_rollout_analysis.py" \
      --rollout-kind student_after_bridge --rollout-root "${rollout}" \
      --analysis-root "${output}" --num-shards 64 \
      --expected-traces 13244 --num-rollouts 8
  fi
  test -f "${final_marker}"
  printf 'task\t%s\nhorizon\t%s\nreplicate\t%s\nmode\t%s\ncompleted_at\t%s\n' \
    "${task_id}" "${h}" "${replicate}" "${mode}" "$(date --iso-8601=seconds)" \
    >"${RUN_ROOT}/status/task_${task_id}.tsv.tmp.${SLURM_JOB_ID}"
  mv "${RUN_ROOT}/status/task_${task_id}.tsv.tmp.${SLURM_JOB_ID}" \
    "${RUN_ROOT}/status/task_${task_id}.tsv"
done

touch "${RUN_ROOT}/WORKER_${WORKER_INDEX}_COMPLETE"

# Whichever worker finishes last performs the three lightweight strict
# aggregates.  This keeps concurrent node usage capped at three and avoids a
# fourth Slurm allocation.
worker_marker_count=$(find "${RUN_ROOT}" -maxdepth 1 -type f -name 'WORKER_*_COMPLETE' | wc -l)
if [[ "${worker_marker_count}" -eq "${TOTAL_WORKERS}" ]] && mkdir "${RUN_ROOT}/aggregate.lock" 2>/dev/null; then
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
  touch "${RUN_ROOT}/SBRIDGE_HORIZON_SCORE_3N_COMPLETE"
fi

echo "SBRIDGE_HORIZON_SCORE_3N_COMPLETE job=${SLURM_JOB_ID} worker=${WORKER_INDEX} node=${SLURMD_NODENAME} time=$(date --iso-8601=seconds)"
