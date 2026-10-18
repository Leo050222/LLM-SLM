#!/usr/bin/env bash
#SBATCH --job-name=verify_tfull_n8
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/dataset/train/teacher_full_qwen3_14b_n8_seed20260826/logs/%j.verify.out
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/dataset/train/teacher_full_qwen3_14b_n8_seed20260826/logs/%j.verify.err

set -euo pipefail
ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
OUTPUT=${ROOT}/dataset/train/teacher_full_qwen3_14b_n8_seed20260826
PYTHON=/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python
test -f "${OUTPUT}/GPU_GENERATION_COMPLETE"
"${PYTHON}" "${ROOT}/ocrd_pipeline/full_rollout/verify_full_rollout_batch.py" \
  --mode teacher_from_anchor \
  --anchor-root "${ROOT}/ocrd_anchor_train25_qwen3_1_7b_nonthinking_seed20260811_transformers_v2/judge_remote_gpt56_luna_batch4" \
  --output-root "${OUTPUT}" \
  --num-shards 8 \
  --num-rollouts 8 \
  --model /storage/yukaichengLab/share/model/Qwen3-14B \
  --max-model-len 40960 \
  --expected-count 13244
