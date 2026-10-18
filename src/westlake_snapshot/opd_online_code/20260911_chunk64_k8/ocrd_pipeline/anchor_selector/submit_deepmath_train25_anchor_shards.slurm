#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH --gres=gpu:H800:1
#SBATCH --array=0-15%8
#SBATCH --exclude=gnho009,gnho018
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH -J anchor14k
#SBATCH -o /storage/yukaichengLab/wangxin/test_eiloeo/ocrd_anchor_train25_qwen3_1_7b_nonthinking_seed20260811_transformers_v2/logs/%A_%a.out
#SBATCH -e /storage/yukaichengLab/wangxin/test_eiloeo/ocrd_anchor_train25_qwen3_1_7b_nonthinking_seed20260811_transformers_v2/logs/%A_%a.err

set -euo pipefail

ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
MODULE=${ROOT}/ocrd_pipeline/anchor_selector
SOURCE=${ROOT}/opd_rollouts/deepmath_ge6_train25pct_qwen3_1_7b_native_nonthinking_seed20260811
RUN_ROOT=${ROOT}/ocrd_anchor_train25_qwen3_1_7b_nonthinking_seed20260811_transformers_v2
PYTHON=/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python
SHARD=$(printf "%02d" "${SLURM_ARRAY_TASK_ID}")
TRACE=${SOURCE}/shards/shard_${SHARD}/rollouts.jsonl
OUT=${RUN_ROOT}/shards/shard_${SHARD}
CACHE_ROOT=/tmp/anchor14k_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=8
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export XDG_CACHE_HOME=${CACHE_ROOT}/xdg
export TORCHINDUCTOR_CACHE_DIR=${CACHE_ROOT}/torchinductor
export TRITON_CACHE_DIR=${CACHE_ROOT}/triton

mkdir -p "${OUT}" "${XDG_CACHE_HOME}" "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"

echo "ANCHOR14K_START job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} node=${SLURMD_NODENAME} trace=${TRACE}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
"${PYTHON}" -c 'import torch; x=torch.randn(256,256,device="cuda"); y=x@x; torch.cuda.synchronize(); print("CUDA_WITNESS_OK", float(y.sum()))'

"${PYTHON}" "${MODULE}/run_anchor_shard.py" teacher-transformers \
  --trace-shard "${TRACE}" \
  --output-dir "${OUT}" \
  --resume \
  --attention-implementation sdpa

# A new Python process is required here so vLLM releases the teacher weights
# before Transformers loads the student model.
"${PYTHON}" "${MODULE}/run_anchor_shard.py" student-candidates \
  --trace-shard "${TRACE}" \
  --output-dir "${OUT}" \
  --resume \
  --attention-implementation sdpa

touch "${OUT}/SHARD_COMPLETE"
echo "ANCHOR14K_COMPLETE job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} node=${SLURMD_NODENAME}"
