#!/usr/bin/env bash
#SBATCH --job-name=bridge256_verify
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/ocrd_anchor_train25_qwen3_1_7b_nonthinking_seed20260811_transformers_v2/bridge_qwen3_14b_h256_n8_seed20260822/logs/%j.verify.out
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/ocrd_anchor_train25_qwen3_1_7b_nonthinking_seed20260811_transformers_v2/bridge_qwen3_14b_h256_n8_seed20260822/logs/%j.verify.err

set -euo pipefail

ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
MODULE=${ROOT}/ocrd_pipeline/bridge_generator
ANCHOR_ROOT=${ROOT}/ocrd_anchor_train25_qwen3_1_7b_nonthinking_seed20260811_transformers_v2/judge_remote_gpt56_luna_batch4
OUTPUT_ROOT=${ROOT}/ocrd_anchor_train25_qwen3_1_7b_nonthinking_seed20260811_transformers_v2/bridge_qwen3_14b_h256_n8_seed20260822
PYTHON=/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python
MODEL=/storage/yukaichengLab/share/model/Qwen3-14B

test -f "${OUTPUT_ROOT}/GPU_GENERATION_COMPLETE"
"${PYTHON}" "${MODULE}/verify_bridge_batch.py" \
  --anchor-root "${ANCHOR_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --num-shards 8 --expected-count 13244 \
  --h 256 --num-rollouts 8 --teacher "${MODEL}"
