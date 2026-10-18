#!/usr/bin/env bash
#SBATCH --job-name=verify_sbridge
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/dataset/train/student_after_bridge_qwen3_1_7b_h256_n8_seed20260826/logs/%j.verify.out
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/dataset/train/student_after_bridge_qwen3_1_7b_h256_n8_seed20260826/logs/%j.verify.err

set -euo pipefail
ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
OUTPUT=${ROOT}/dataset/train/student_after_bridge_qwen3_1_7b_h256_n8_seed20260826
PYTHON=/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python
test -f "${OUTPUT}/GPU_GENERATION_COMPLETE"
"${PYTHON}" "${ROOT}/ocrd_pipeline/full_rollout/verify_full_rollout_batch.py" \
  --mode student_after_bridge \
  --bridge-manifest "${ROOT}/dataset/train/anchor_bridge_qwen3_14b_h256_n8/manifest.jsonl" \
  --output-root "${OUTPUT}" \
  --num-shards 8 \
  --num-rollouts 8 \
  --model /storage/yukaichengLab/share/model/Qwen3-1.7B \
  --max-model-len 40960 \
  --expected-count 13244
