#!/usr/bin/env bash
#SBATCH --job-name=smk_sprefix_n8
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --gres=gpu:H800:1
#SBATCH --exclude=gnho006,gnho009,gnho017,gnho018,gnho036
#SBATCH --time=01:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/dataset/train/student_from_anchor_prefix_qwen3_1_7b_n8_seed20260827_smoke4/logs/%j.out
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/dataset/train/student_from_anchor_prefix_qwen3_1_7b_n8_seed20260827_smoke4/logs/%j.err

set -euo pipefail
ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
OUTPUT=${ROOT}/dataset/train/student_from_anchor_prefix_qwen3_1_7b_n8_seed20260827_smoke4
ANCHOR_ROOT=${ROOT}/ocrd_anchor_train25_qwen3_1_7b_nonthinking_seed20260811_transformers_v2/judge_remote_gpt56_luna_batch4
PYTHON=/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python
mkdir -p "${OUTPUT}/logs"
test ! -e "${OUTPUT}/FULL_ROLLOUT_BATCH_COMPLETE"
bash "${ROOT}/ocrd_pipeline/full_rollout/run_full_rollout_worker.sh" \
  student_from_anchor_prefix 0 1 "${OUTPUT}" 4
"${PYTHON}" "${ROOT}/ocrd_pipeline/full_rollout/verify_full_rollout_batch.py" \
  --mode student_from_anchor_prefix \
  --anchor-root "${ANCHOR_ROOT}" \
  --output-root "${OUTPUT}" \
  --num-shards 1 \
  --max-records 4 \
  --num-rollouts 8 \
  --model /storage/yukaichengLab/share/model/Qwen3-1.7B \
  --max-model-len 40960 \
  --seed 20260827 \
  --expected-count 4
echo STUDENT_FROM_ANCHOR_PREFIX_SMOKE_COMPLETE
