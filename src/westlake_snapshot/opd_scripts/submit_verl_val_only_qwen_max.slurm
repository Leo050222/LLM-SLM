#!/bin/bash
#SBATCH --job-name=opd_val_max
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:H800:4
#SBATCH --time=03:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/opd_val_max.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/opd_val_max.%j.err

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
export MAX_RESPONSE_LENGTH=38912
export MAX_MODEL_LEN=40960
export GPU_MEMORY_UTILIZATION=0.65
export TRITON_CACHE_DIR="/tmp/triton_cache_opd_val_max_${SLURM_JOB_ID}"
export TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor_opd_val_max_${SLURM_JOB_ID}"

RUN=/storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/run_verl_val_only_qwen.sh
OUT=/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval/verl_val_only_max_512

mkdir -p "${OUT}"
mkdir -p "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start time: $(date --iso-8601=seconds)"
echo "Lengths: prompt=${MAX_PROMPT_LENGTH} response=${MAX_RESPONSE_LENGTH} model=${MAX_MODEL_LEN}"
echo "vLLM GPU memory utilization: ${GPU_MEMORY_UTILIZATION}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

run_if_missing() {
  local model_path=$1
  local model_name=$2
  local output_dir=$3
  local output_file="${output_dir}/0.jsonl"
  if [[ -f "${output_file}" ]] && [[ "$(wc -l < "${output_file}")" -eq 512 ]]; then
    echo "Skipping completed ${model_name}: ${output_file}"
    return 0
  fi
  bash "${RUN}" "${model_path}" "${model_name}" "${output_dir}"
}

run_if_missing \
  /storage/yukaichengLab/share/model/Qwen3-4B \
  qwen3_4b_base_response38912 \
  "${OUT}/qwen3_4b_base"

run_if_missing \
  /storage/yukaichengLab/wangxin/test_eiloeo/opd_checkpoints/qwen3_4b_from_14b_gkd_gsm8k_math/global_step_234/actor/huggingface \
  qwen3_4b_opd_step234_response38912 \
  "${OUT}/qwen3_4b_opd_step234"

run_if_missing \
  /storage/yukaichengLab/share/model/Qwen3-14B \
  qwen3_14b_base_response38912 \
  "${OUT}/qwen3_14b_base"

echo "End time: $(date --iso-8601=seconds)"
