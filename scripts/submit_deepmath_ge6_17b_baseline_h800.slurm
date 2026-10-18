#!/bin/bash
#SBATCH --job-name=dm_ge6_17base
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --gres=gpu:H800:1
#SBATCH --array=0-3
#SBATCH --time=12:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/dm_ge6_17base.%A_%a.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/dm_ge6_17base.%A_%a.err

set -euo pipefail
source /storage/yukaichengLab/wangxin/miniconda3/etc/profile.d/conda.sh
conda activate verl-opd

ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
TASK=${SLURM_ARRAY_TASK_ID}
DATA=${ROOT}/opd_data/deepmath_ge6_opd_train15pct_seed20260811/eval_shards4/test_1000.shard${TASK}.parquet
OUT=${ROOT}/opd_eval_deepmath/ge6_test1000_qwen3_1_7b_baseline_seed20260811/shards/shard_${TASK}
SCRIPT=${ROOT}/opd_scripts/run_opd_logit_diagnostic.py

export PYTHONPATH=${ROOT}/verl
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=8 VLLM_WORKER_MULTIPROC_METHOD=spawn PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_PORT=$((21000 + (SLURM_ARRAY_JOB_ID % 500) * 8 + TASK))
export TRITON_CACHE_DIR=/tmp/triton_dm_ge6_17base_${SLURM_ARRAY_JOB_ID}_${TASK}
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_dm_ge6_17base_${SLURM_ARRAY_JOB_ID}_${TASK}
mkdir -p "${OUT}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"

echo "BASELINE_ARRAY=${SLURM_ARRAY_JOB_ID} TASK=${TASK} NODE=${SLURMD_NODENAME} START=$(date --iso-8601=seconds)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
python "${SCRIPT}" generate --output-dir "${OUT}" \
  --student /storage/yukaichengLab/share/model/Qwen3-1.7B \
  --teacher /storage/yukaichengLab/share/model/Qwen3-14B \
  --dataset-name deepmath_ge6_test1000 --data "${DATA}" \
  --prompt-style deepmath --scorer math --score-timeout 20 \
  --seed 20260811 --max-tokens 40264 --max-model-len 40960 --topk 16
echo "DEEPMATH_GE6_17B_BASELINE_SHARD_OK task=${TASK} $(date --iso-8601=seconds)"
