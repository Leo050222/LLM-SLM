#!/bin/bash
#SBATCH --job-name=opd_val_exact
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:H800:4
#SBATCH --time=01:30:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/opd_val_exact.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/opd_val_exact.%j.err

set -euo pipefail

source /storage/yukaichengLab/wangxin/miniconda3/etc/profile.d/conda.sh
conda activate verl-opd

export PYTHONPATH=/storage/yukaichengLab/wangxin/test_eiloeo/verl
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=8

RUN=/storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/run_verl_val_only_qwen.sh
OUT=/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval/verl_val_only_512

mkdir -p "${OUT}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start time: $(date --iso-8601=seconds)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

bash "${RUN}" \
  /storage/yukaichengLab/share/model/Qwen3-4B \
  qwen3_4b_base \
  "${OUT}/qwen3_4b_base"

bash "${RUN}" \
  /storage/yukaichengLab/wangxin/test_eiloeo/opd_checkpoints/qwen3_4b_from_14b_gkd_gsm8k_math/global_step_234/actor/huggingface \
  qwen3_4b_opd_step234 \
  "${OUT}/qwen3_4b_opd_step234"

bash "${RUN}" \
  /storage/yukaichengLab/share/model/Qwen3-14B \
  qwen3_14b_base \
  "${OUT}/qwen3_14b_base"

echo "End time: $(date --iso-8601=seconds)"
