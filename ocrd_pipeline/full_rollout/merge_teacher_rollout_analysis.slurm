#!/usr/bin/env bash
#SBATCH --job-name=merge_tfull_stats
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/dataset/train/teacher_full_qwen3_14b_n8_seed20260826/analysis_correctness_length/logs/%j.merge.out
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/dataset/train/teacher_full_qwen3_14b_n8_seed20260826/analysis_correctness_length/logs/%j.merge.err

set -euo pipefail
ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
ROLLOUT=${ROOT}/dataset/train/teacher_full_qwen3_14b_n8_seed20260826
OUTPUT=${ROLLOUT}/analysis_correctness_length
PYTHON=/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python

test -f "${ROLLOUT}/FULL_ROLLOUT_BATCH_COMPLETE"
"${PYTHON}" "${ROOT}/ocrd_pipeline/full_rollout/merge_teacher_rollout_analysis.py" \
  --rollout-root "${ROLLOUT}" \
  --analysis-root "${OUTPUT}" \
  --num-shards 64 \
  --expected-traces 13244 \
  --num-rollouts 8
