#!/bin/bash
#SBATCH --job-name=dm_max_14b
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --nodelist=gnho020
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:H800:1
#SBATCH --array=0-3
#SBATCH --time=48:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/dm_max_14b.%A_%a.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/dm_max_14b.%A_%a.err

set -euo pipefail
source /storage/yukaichengLab/wangxin/miniconda3/etc/profile.d/conda.sh
conda activate verl-opd

ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
SCRIPT=${ROOT}/opd_scripts/run_opd_logit_diagnostic.py
DATA_ROOT=${ROOT}/opd_data/deepmath_103k
OUTPUT=${ROOT}/opd_eval_deepmath/maxlen_seed20260809
TASK=${SLURM_ARRAY_TASK_ID}
data=${DATA_ROOT}/sample_1000.shard${TASK}.parquet
out=${OUTPUT}/qwen3_14b_shards/shard_${TASK}

export PYTHONPATH=${ROOT}/verl
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=8 VLLM_WORKER_MULTIPROC_METHOD=spawn PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_PORT=$((20000 + (SLURM_ARRAY_JOB_ID % 1000) * 16 + TASK))
export TRITON_CACHE_DIR=/tmp/triton_dm_max_14b_${SLURM_ARRAY_JOB_ID}_${TASK}
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_dm_max_14b_${SLURM_ARRAY_JOB_ID}_${TASK}
mkdir -p "${out}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"
echo "ARRAY_JOB=${SLURM_ARRAY_JOB_ID} TASK=${TASK} VLLM_PORT=${VLLM_PORT} NODE=${SLURMD_NODENAME} START=$(date --iso-8601=seconds)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

python "${SCRIPT}" generate --output-dir "${out}" \
  --student /storage/yukaichengLab/share/model/Qwen3-14B \
  --teacher /storage/yukaichengLab/share/model/Qwen3-14B \
  --dataset-name deepmath_103k --data "${data}" --prompt-style deepmath --scorer math \
  --score-timeout 20 --seed 20260809 --max-tokens 40512 --max-model-len 40960 --topk 128
echo "MAXLEN_14B_SHARD_COMPLETE=${TASK} $(date --iso-8601=seconds)"
