#!/usr/bin/env bash
#SBATCH --job-name=teacher_full_n8
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --gres=gpu:H800:8
#SBATCH --exclusive
#SBATCH --exclude=gnho009,gnho018,gnho036
#SBATCH --time=20:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/dataset/train/teacher_full_qwen3_14b_n8_seed20260826/logs/%j.out
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/dataset/train/teacher_full_qwen3_14b_n8_seed20260826/logs/%j.err

set -euo pipefail
ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
OUTPUT=${ROOT}/dataset/train/teacher_full_qwen3_14b_n8_seed20260826
mkdir -p "${OUTPUT}/logs"
bash "${ROOT}/ocrd_pipeline/full_rollout/run_full_rollout_node8.sh" \
  teacher_from_anchor "${OUTPUT}"
