#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 || $# -gt 6 ]]; then
  echo "usage: $0 SHARD_ID NUM_SHARDS OUTPUT_ROOT BASE_SEED MAX_RECORDS_OR_ZERO [BRIDGE_TOKEN_LIMIT_OR_ZERO]" >&2
  exit 2
fi

SHARD_ID=$1
NUM_SHARDS=$2
OUTPUT_ROOT=$3
BASE_SEED=$4
MAX_RECORDS=$5
BRIDGE_TOKEN_LIMIT=${6:-0}
ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
MODULE=${ROOT}/ocrd_pipeline/full_rollout
BRIDGE_MANIFEST=${ROOT}/dataset/train/anchor_bridge_qwen3_14b_h256_n8/manifest.jsonl
MODEL=/storage/yukaichengLab/share/model/Qwen3-1.7B
PYTHON=/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python

[[ "${SHARD_ID}" =~ ^[0-9]+$ ]]
[[ "${NUM_SHARDS}" =~ ^[0-9]+$ ]]
[[ "${BASE_SEED}" =~ ^[0-9]+$ ]]
[[ "${MAX_RECORDS}" =~ ^[0-9]+$ ]]
[[ "${BRIDGE_TOKEN_LIMIT}" =~ ^[0-9]+$ ]]

CACHE_ROOT=/tmp/ocrd_sbridge_replicate_${SLURM_JOB_ID}_${BASE_SEED}_${SHARD_ID}
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_PORT=$((22000 + (SLURM_JOB_ID % 1000) * 8 + SHARD_ID))
export XDG_CACHE_HOME=${CACHE_ROOT}/xdg
export VLLM_CACHE_ROOT=${CACHE_ROOT}/vllm
export TORCHINDUCTOR_CACHE_DIR=${CACHE_ROOT}/torchinductor
export TRITON_CACHE_DIR=${CACHE_ROOT}/triton
mkdir -p "${XDG_CACHE_HOME}" "${VLLM_CACHE_ROOT}" \
  "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"

echo "STUDENT_BRIDGE_REPLICATE_WORKER_START job=${SLURM_JOB_ID} seed=${BASE_SEED} shard=${SHARD_ID}/${NUM_SHARDS} node=${SLURMD_NODENAME} visible=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
"${PYTHON}" -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 1; free,total=torch.cuda.mem_get_info(); used=total-free; assert used < 1024*1024**2, f"visible GPU is not clean after CUDA context creation: used={used/2**20:.1f} MiB"; x=torch.arange(8,device="cuda"); assert int(x.sum()) == 28; print("STUDENT_BRIDGE_REPLICATE_CUDA_WITNESS_OK",torch.cuda.get_device_name(),f"used_mib={used/2**20:.1f}")'

MAX_ARGS=()
if [[ "${MAX_RECORDS}" -gt 0 ]]; then
  MAX_ARGS=(--max-records "${MAX_RECORDS}")
fi
BRIDGE_ARGS=()
if [[ "${BRIDGE_TOKEN_LIMIT}" -gt 0 ]]; then
  BRIDGE_ARGS=(--bridge-token-limit "${BRIDGE_TOKEN_LIMIT}")
fi

"${PYTHON}" "${MODULE}/run_full_rollout_batch.py" \
  --mode student_after_bridge \
  --bridge-manifest "${BRIDGE_MANIFEST}" \
  --output-root "${OUTPUT_ROOT}" \
  --shard-id "${SHARD_ID}" \
  --num-shards "${NUM_SHARDS}" \
  "${MAX_ARGS[@]}" \
  --model "${MODEL}" \
  --tokenizer "${MODEL}" \
  --temperature 1.0 \
  --top-p 1.0 \
  --top-k -1 \
  --seed "${BASE_SEED}" \
  --first-step-logprobs 16 \
  --max-model-len 40960 \
  --num-rollouts 8 \
  "${BRIDGE_ARGS[@]}" \
  --trace-batch-size 4 \
  --gpu-memory-utilization 0.90

echo "STUDENT_BRIDGE_REPLICATE_WORKER_COMPLETE job=${SLURM_JOB_ID} seed=${BASE_SEED} shard=${SHARD_ID}/${NUM_SHARDS} node=${SLURMD_NODENAME}"
