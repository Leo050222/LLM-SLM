#!/usr/bin/env bash
#SBATCH --job-name=verify_sbridge_eos
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/dataset/train/student_after_bridge_qwen3_1_7b_h256_n8_seed20260826/logs/%j.verify_refresh.out
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/dataset/train/student_after_bridge_qwen3_1_7b_h256_n8_seed20260826/logs/%j.verify_refresh.err

set -euo pipefail
ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
ROLLOUT=${ROOT}/dataset/train/student_after_bridge_qwen3_1_7b_h256_n8_seed20260826
ANALYSIS=${ROLLOUT}/analysis_correctness_length_single_job
PYTHON=/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python
VERIFIER=${ROOT}/ocrd_pipeline/full_rollout/verify_full_rollout_batch.py
MERGER=${ROOT}/ocrd_pipeline/full_rollout/merge_teacher_rollout_analysis.py
BRIDGES=${ROOT}/dataset/train/anchor_bridge_qwen3_14b_h256_n8/manifest.jsonl
MODEL=/storage/yukaichengLab/share/model/Qwen3-1.7B

test -f "${ROLLOUT}/GPU_GENERATION_COMPLETE"
test "$(find "${ROLLOUT}/shards" -type f -name SHARD_COMPLETE | wc -l)" -eq 8
trap 'rc=$?; echo "STUDENT_AFTER_BRIDGE_VERIFY_REFRESH_ERROR job=${SLURM_JOB_ID:-unset} line=${LINENO} command=${BASH_COMMAND} exit_code=${rc}" >&2' ERR

common_args=(
  --mode student_after_bridge
  --bridge-manifest "${BRIDGES}"
  --output-root "${ROLLOUT}"
  --num-shards 8
  --num-rollouts 8
  --model "${MODEL}"
  --max-model-len 40960
  --expected-count 13244
)

# First pass is strictly read-only. Only a clean pass permits the formal
# verifier to atomically publish the manifest, summary, and completion marker.
"${PYTHON}" "${VERIFIER}" "${common_args[@]}" --check-only
"${PYTHON}" "${VERIFIER}" "${common_args[@]}"
test -f "${ROLLOUT}/FULL_ROLLOUT_BATCH_COMPLETE"

# The score shards already exist. Re-merge them so the analysis summary records
# that its source now has strict verification, without recomputing any scores.
"${PYTHON}" "${MERGER}" \
  --rollout-kind student_after_bridge \
  --rollout-root "${ROLLOUT}" \
  --analysis-root "${ANALYSIS}" \
  --num-shards 64 \
  --expected-traces 13244 \
  --num-rollouts 8
"${PYTHON}" -c 'import json, pathlib, sys; p=pathlib.Path(sys.argv[1]); d=json.loads(p.read_text()); assert d["source_strict_verification_complete"] is True' "${ANALYSIS}/summary.json"
echo "STUDENT_AFTER_BRIDGE_MULTIEOS_VERIFY_AND_REFRESH_COMPLETE job=${SLURM_JOB_ID}"
