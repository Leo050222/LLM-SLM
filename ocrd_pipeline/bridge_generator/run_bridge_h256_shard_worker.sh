#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 SHARD_ID NUM_SHARDS" >&2
  exit 2
fi

SHARD_ID=$1
NUM_SHARDS=$2
ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
MODULE=${ROOT}/ocrd_pipeline/bridge_generator
ANCHOR_ROOT=${ROOT}/ocrd_anchor_train25_qwen3_1_7b_nonthinking_seed20260811_transformers_v2/judge_remote_gpt56_luna_batch4
OUTPUT_ROOT=${ROOT}/ocrd_anchor_train25_qwen3_1_7b_nonthinking_seed20260811_transformers_v2/bridge_qwen3_14b_h256_n8_seed20260822
PYTHON=/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python
MODEL=/storage/yukaichengLab/share/model/Qwen3-14B
CACHE_ROOT=/tmp/ocrd_bridge_h256_${SLURM_JOB_ID}_${SHARD_ID}

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=6
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_PORT=$((24000 + (SLURM_JOB_ID % 1000) * 8 + SHARD_ID))
export XDG_CACHE_HOME=${CACHE_ROOT}/xdg
export VLLM_CACHE_ROOT=${CACHE_ROOT}/vllm
export TORCHINDUCTOR_CACHE_DIR=${CACHE_ROOT}/torchinductor
export TRITON_CACHE_DIR=${CACHE_ROOT}/triton

mkdir -p "${XDG_CACHE_HOME}" "${VLLM_CACHE_ROOT}" \
  "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"

echo "BRIDGE_WORKER_START job=${SLURM_JOB_ID} shard=${SHARD_ID}/${NUM_SHARDS} node=${SLURMD_NODENAME} visible=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
"${PYTHON}" -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 1; x=torch.arange(8,device="cuda"); assert int(x.sum()) == 28; print("BRIDGE_CUDA_WITNESS_OK",torch.cuda.get_device_name())'

"${PYTHON}" "${MODULE}/run_bridge_batch.py" \
  --anchor-root "${ANCHOR_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --shard-id "${SHARD_ID}" \
  --num-shards "${NUM_SHARDS}" \
  --h 256 \
  --teacher "${MODEL}" \
  --temperature 1.0 \
  --top-p 1.0 \
  --top-k 0 \
  --seed 20260822 \
  --first-step-logprobs 16 \
  --max-model-len 4096 \
  --num-rollouts 8 \
  --trace-batch-size 8 \
  --gpu-memory-utilization 0.90

echo "BRIDGE_WORKER_COMPLETE job=${SLURM_JOB_ID} shard=${SHARD_ID}/${NUM_SHARDS} node=${SLURMD_NODENAME}"
