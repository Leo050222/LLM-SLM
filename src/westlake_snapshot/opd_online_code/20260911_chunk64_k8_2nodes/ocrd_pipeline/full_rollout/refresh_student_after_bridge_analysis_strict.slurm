#!/usr/bin/env bash
#SBATCH --job-name=refresh_sbridge_strict
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/dataset/train/student_after_bridge_qwen3_1_7b_h256_n8_seed20260826/logs/%j.refresh_analysis.out
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/dataset/train/student_after_bridge_qwen3_1_7b_h256_n8_seed20260826/logs/%j.refresh_analysis.err

set -euo pipefail
ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
ROLLOUT=${ROOT}/dataset/train/student_after_bridge_qwen3_1_7b_h256_n8_seed20260826
ANALYSIS=${ROLLOUT}/analysis_correctness_length_single_job
PYTHON=/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python
MERGER=${ROOT}/ocrd_pipeline/full_rollout/merge_teacher_rollout_analysis.py

test -f "${ROLLOUT}/FULL_ROLLOUT_BATCH_COMPLETE"
test -f "${ROLLOUT}/summary.json"
test -f "${ROLLOUT}/manifest.jsonl"
test -f "${ANALYSIS}/STUDENT_AFTER_BRIDGE_ROLLOUT_ANALYSIS_COMPLETE"
trap 'rc=$?; echo "STUDENT_AFTER_BRIDGE_ANALYSIS_REFRESH_ERROR job=${SLURM_JOB_ID:-unset} line=${LINENO} command=${BASH_COMMAND} exit_code=${rc}" >&2' ERR

"${PYTHON}" "${MERGER}" \
  --rollout-kind student_after_bridge \
  --rollout-root "${ROLLOUT}" \
  --analysis-root "${ANALYSIS}" \
  --num-shards 64 \
  --expected-traces 13244 \
  --num-rollouts 8 \
  --refresh-completed-summary
"${PYTHON}" -c 'import json, pathlib, sys; p=pathlib.Path(sys.argv[1]); d=json.loads(p.read_text()); assert d["source_strict_verification_complete"] is True' "${ANALYSIS}/summary.json"
echo "STUDENT_AFTER_BRIDGE_ANALYSIS_STRICT_REFRESH_COMPLETE job=${SLURM_JOB_ID}"
