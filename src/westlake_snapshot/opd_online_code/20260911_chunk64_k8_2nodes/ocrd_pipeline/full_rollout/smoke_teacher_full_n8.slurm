#!/usr/bin/env bash
#SBATCH --job-name=smk_tfull_n8
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gres=gpu:H800:1
#SBATCH --exclude=gnho009,gnho018,gnho036
#SBATCH --time=01:30:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/dataset/train/teacher_full_qwen3_14b_n8_seed20260826_smoke4/logs/%j.out
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/dataset/train/teacher_full_qwen3_14b_n8_seed20260826_smoke4/logs/%j.err

set -euo pipefail
ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
OUTPUT=${ROOT}/dataset/train/teacher_full_qwen3_14b_n8_seed20260826_smoke4
PYTHON=/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python
mkdir -p "${OUTPUT}/logs"
test ! -e "${OUTPUT}/FULL_ROLLOUT_BATCH_COMPLETE"
bash "${ROOT}/ocrd_pipeline/full_rollout/run_full_rollout_worker.sh" \
  teacher_from_anchor 0 1 "${OUTPUT}" 4
"${PYTHON}" "${ROOT}/ocrd_pipeline/full_rollout/verify_full_rollout_batch.py" \
  --mode teacher_from_anchor \
  --anchor-root "${ROOT}/ocrd_anchor_train25_qwen3_1_7b_nonthinking_seed20260811_transformers_v2/judge_remote_gpt56_luna_batch4" \
  --output-root "${OUTPUT}" \
  --num-shards 1 \
  --max-records 4 \
  --num-rollouts 8 \
  --model /storage/yukaichengLab/share/model/Qwen3-14B \
  --max-model-len 40960 \
  --expected-count 4
echo TEACHER_FULL_SMOKE_COMPLETE
