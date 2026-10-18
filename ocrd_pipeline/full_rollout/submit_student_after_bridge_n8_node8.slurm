#!/usr/bin/env bash
#SBATCH --job-name=student_bridge_n8
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=384G
#SBATCH --gres=gpu:H800:8
#SBATCH --exclusive
#SBATCH --exclude=gnho006,gnho009,gnho017,gnho018,gnho036
#SBATCH --time=08:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/dataset/train/student_after_bridge_qwen3_1_7b_h256_n8_seed20260826/logs/%j.out
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/dataset/train/student_after_bridge_qwen3_1_7b_h256_n8_seed20260826/logs/%j.err

set -euo pipefail
ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
OUTPUT=${ROOT}/dataset/train/student_after_bridge_qwen3_1_7b_h256_n8_seed20260826
mkdir -p "${OUTPUT}/logs"
trap 'rc=$?; echo "STUDENT_ROLLOUT_SLURM_ERROR job=${SLURM_JOB_ID:-unset} line=${LINENO} command=${BASH_COMMAND} exit_code=${rc}" >&2' ERR
bash "${ROOT}/ocrd_pipeline/full_rollout/run_full_rollout_node8.sh" \
  student_after_bridge "${OUTPUT}"
