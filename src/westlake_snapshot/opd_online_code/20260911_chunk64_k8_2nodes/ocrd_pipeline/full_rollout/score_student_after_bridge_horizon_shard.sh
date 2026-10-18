#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 BRIDGE_TOKEN_LIMIT REPLICATE SHARD_INDEX RUN_ROOT" >&2
  exit 2
fi

H=$1
REPLICATE=$2
SHARD_INDEX=$3
RUN_ROOT=$4
[[ "${H}" =~ ^(32|64|128)$ ]]
[[ "${REPLICATE}" =~ ^[0-7]$ ]]
[[ "${SHARD_INDEX}" =~ ^([0-9]|[1-5][0-9]|6[0-3])$ ]]

ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
PYTHON=/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python
DATA=${ROOT}/dataset/train/original_deepmath_ge6_train25pct/train_unique.parquet
PARENT=${ROOT}/dataset/train/student_after_bridge_qwen3_1_7b_h${H}_n8_x8_seed20260826
ROLLOUT=${PARENT}/$(printf 'replicate_%02d' "${REPLICATE}")
OUTPUT=${ROLLOUT}/analysis_correctness_length_single_job
MARKER=${OUTPUT}/shards/$(printf 'shard_%03d' "${SHARD_INDEX}")/SCORE_SHARD_COMPLETE
TASK_ID=$(( (${H} == 32 ? 0 : ${H} == 64 ? 8 : 16) + REPLICATE ))

[[ ! -f "${MARKER}" ]] || exit 0
mkdir -p "${RUN_ROOT}/logs"
"${PYTHON}" "${ROOT}/ocrd_pipeline/full_rollout/analyze_teacher_rollout_shard.py" \
  --rollout-kind student_after_bridge --rollout-root "${ROLLOUT}" \
  --dataset "${DATA}" --output-root "${OUTPUT}" \
  --shard-index "${SHARD_INDEX}" --num-shards 64 \
  --num-rollouts 8 --score-timeout 20 \
  >"${RUN_ROOT}/logs/task_${TASK_ID}_shard_${SHARD_INDEX}.out" \
  2>"${RUN_ROOT}/logs/task_${TASK_ID}_shard_${SHARD_INDEX}.err"
test -f "${MARKER}"
