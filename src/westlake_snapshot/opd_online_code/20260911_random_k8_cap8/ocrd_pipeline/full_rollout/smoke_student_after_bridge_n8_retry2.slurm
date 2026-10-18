#!/usr/bin/env bash
#SBATCH --job-name=smk_sbridge_r2
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --gres=gpu:H800:1
#SBATCH --exclude=gnho006,gnho009,gnho018,gnho036
#SBATCH --time=01:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/dataset/train/student_after_bridge_qwen3_1_7b_h256_n8_seed20260826_smoke4_retry2/logs/%j.out
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/dataset/train/student_after_bridge_qwen3_1_7b_h256_n8_seed20260826_smoke4_retry2/logs/%j.err

set -euo pipefail
ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
OUTPUT=${ROOT}/dataset/train/student_after_bridge_qwen3_1_7b_h256_n8_seed20260826_smoke4_retry2
PYTHON=/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python
mkdir -p "${OUTPUT}/logs"
test ! -e "${OUTPUT}/FULL_ROLLOUT_BATCH_COMPLETE"
bash "${ROOT}/ocrd_pipeline/full_rollout/run_full_rollout_worker.sh" \
  student_after_bridge 0 1 "${OUTPUT}" 4
"${PYTHON}" "${ROOT}/ocrd_pipeline/full_rollout/verify_full_rollout_batch.py" \
  --mode student_after_bridge \
  --bridge-manifest "${ROOT}/dataset/train/anchor_bridge_qwen3_14b_h256_n8/manifest.jsonl" \
  --output-root "${OUTPUT}" \
  --num-shards 1 \
  --max-records 4 \
  --num-rollouts 8 \
  --model /storage/yukaichengLab/share/model/Qwen3-1.7B \
  --max-model-len 40960 \
  --expected-count 4
echo STUDENT_AFTER_BRIDGE_RETRY2_SMOKE_COMPLETE
