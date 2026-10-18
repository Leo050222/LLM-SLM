#!/bin/bash
#SBATCH --job-name=dm_ge6_ckpteval_h800
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --gres=gpu:H800:1
#SBATCH --exclude=gnho018
#SBATCH --array=0-15
#SBATCH --time=12:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/dm_ge6_ckpteval_h800.%A_%a.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/dm_ge6_ckpteval_h800.%A_%a.err

set -euo pipefail
source /storage/yukaichengLab/wangxin/miniconda3/etc/profile.d/conda.sh
conda activate verl-opd

ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
STEPS=(27 80 106 133)
TASK=${SLURM_ARRAY_TASK_ID}
CHECKPOINT_INDEX=$((TASK / 4))
SHARD=$((TASK % 4))
STEP=${STEPS[${CHECKPOINT_INDEX}]}
DATA=${ROOT}/opd_data/deepmath_ge6_opd_train15pct_seed20260811/eval_shards4/test_1000.shard${SHARD}.parquet
RUN=${ROOT}/opd_eval_deepmath/ge6_test1000_qwen3_1_7b_top16_rkl_step${STEP}_seed20260811
OUT=${RUN}/shards/shard_${SHARD}
SCRIPT=${ROOT}/opd_scripts/run_opd_logit_diagnostic_tokenizer.py
MODEL=${ROOT}/opd_checkpoints/qwen3_1_7b_from_14b_deepmath_ge6_train15pct_student_top16_rkl_16h800/global_step_${STEP}/actor/huggingface

export PYTHONPATH=${ROOT}/verl
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=8 VLLM_WORKER_MULTIPROC_METHOD=spawn PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_PORT=$((28000 + (SLURM_ARRAY_JOB_ID % 400) * 32 + TASK))
export TRITON_CACHE_DIR=/tmp/triton_dm_ge6_ckpteval_${SLURM_ARRAY_JOB_ID}_${TASK}
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_dm_ge6_ckpteval_${SLURM_ARRAY_JOB_ID}_${TASK}

if [[ ! -s "${MODEL}/config.json" || ! -s "${MODEL}/model.safetensors.index.json" ]]; then
  echo "Incomplete step${STEP} Hugging Face checkpoint: ${MODEL}" >&2
  exit 2
fi
if [[ -e "${OUT}/rollouts.jsonl" || -e "${OUT}/run_config.json" ]]; then
  echo "Refusing to overwrite existing shard output: ${OUT}" >&2
  exit 3
fi

echo "CHECKPOINT_EVAL_H800_ARRAY=${SLURM_ARRAY_JOB_ID} TASK=${TASK} STEP=${STEP} SHARD=${SHARD} NODE=${SLURMD_NODENAME} START=$(date --iso-8601=seconds)"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
python -c 'import torch; assert torch.cuda.is_available(); x=torch.arange(8,device="cuda"); print("CUDA_PREFLIGHT_OK",torch.cuda.get_device_name(),int(x.sum()))'
mkdir -p "${OUT}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"
python "${SCRIPT}" generate --output-dir "${OUT}" \
  --student "${MODEL}" \
  --tokenizer /storage/yukaichengLab/share/model/Qwen3-1.7B \
  --teacher /storage/yukaichengLab/share/model/Qwen3-14B \
  --dataset-name deepmath_ge6_test1000 --data "${DATA}" \
  --prompt-style deepmath --scorer math --score-timeout 20 \
  --seed 20260811 --max-tokens 40264 --max-model-len 40960 --topk 16
echo "DEEPMATH_GE6_TOP16_RKL_STEP${STEP}_H800_SHARD_OK shard=${SHARD} $(date --iso-8601=seconds)"
