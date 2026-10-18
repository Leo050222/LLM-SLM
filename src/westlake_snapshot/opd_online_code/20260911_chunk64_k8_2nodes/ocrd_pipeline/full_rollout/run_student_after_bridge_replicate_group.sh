#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "usage: $0 PARENT_ROOT FIRST_REPLICATE LAST_REPLICATE MAX_RECORDS_OR_ZERO [BRIDGE_TOKEN_LIMIT_OR_ZERO]" >&2
  exit 2
fi

PARENT_ROOT=$1
FIRST=$2
LAST=$3
MAX_RECORDS=$4
BRIDGE_TOKEN_LIMIT=${5:-0}
ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
BASE_EXPERIMENT_SEED=20260826

for replicate in $(seq "${FIRST}" "${LAST}"); do
  replicate_name=$(printf 'replicate_%02d' "${replicate}")
  output_root=${PARENT_ROOT}/${replicate_name}
  base_seed=$((BASE_EXPERIMENT_SEED + replicate * 1000000))
  mkdir -p "${output_root}/logs"
  printf '%s\n' "${base_seed}" >"${output_root}/BASE_SEED"
  printf '%s\n' "${replicate}" >"${output_root}/REPLICATE_INDEX"
  echo "STUDENT_BRIDGE_REPLICATE_GROUP_ITEM_START replicate=${replicate} seed=${base_seed} node=${SLURMD_NODENAME}"
  bash "${ROOT}/ocrd_pipeline/full_rollout/run_student_after_bridge_replicate_node8.sh" \
    "${output_root}" "${base_seed}" "${MAX_RECORDS}" "${BRIDGE_TOKEN_LIMIT}"
  echo "STUDENT_BRIDGE_REPLICATE_GROUP_ITEM_COMPLETE replicate=${replicate} seed=${base_seed} node=${SLURMD_NODENAME}"
done
