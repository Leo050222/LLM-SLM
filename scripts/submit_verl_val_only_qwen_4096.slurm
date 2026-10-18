#!/bin/bash
#SBATCH --job-name=opd_val_4096
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:H800:4
#SBATCH --time=02:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/opd_val_4096.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/opd_val_4096.%j.err

set -euo pipefail

source /storage/yukaichengLab/wangxin/miniconda3/etc/profile.d/conda.sh
conda activate verl-opd

export PYTHONPATH=/storage/yukaichengLab/wangxin/test_eiloeo/verl
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=8
export MAX_PROMPT_LENGTH=2048
export MAX_RESPONSE_LENGTH=4096
export MAX_MODEL_LEN=6145

RUN=/storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/run_verl_val_only_qwen.sh
OUT=/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval/verl_val_only_4096_512

mkdir -p "${OUT}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start time: $(date --iso-8601=seconds)"
echo "Lengths: prompt=${MAX_PROMPT_LENGTH} response=${MAX_RESPONSE_LENGTH} model=${MAX_MODEL_LEN}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

bash "${RUN}" \
  /storage/yukaichengLab/share/model/Qwen3-4B \
  qwen3_4b_base_response4096 \
  "${OUT}/qwen3_4b_base"

bash "${RUN}" \
  /storage/yukaichengLab/wangxin/test_eiloeo/opd_checkpoints/qwen3_4b_from_14b_gkd_gsm8k_math/global_step_234/actor/huggingface \
  qwen3_4b_opd_step234_response4096 \
  "${OUT}/qwen3_4b_opd_step234"

bash "${RUN}" \
  /storage/yukaichengLab/share/model/Qwen3-14B \
  qwen3_14b_base_response4096 \
  "${OUT}/qwen3_14b_base"

echo "End time: $(date --iso-8601=seconds)"
