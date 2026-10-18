#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "usage: $0 MODE SHARD_ID NUM_SHARDS OUTPUT_ROOT [MAX_RECORDS]" >&2
  exit 2
fi

MODE=$1
SHARD_ID=$2
NUM_SHARDS=$3
OUTPUT_ROOT=$4
MAX_RECORDS=${5:-}
ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
MODULE=${ROOT}/ocrd_pipeline/full_rollout
ANCHOR_ROOT=${ROOT}/ocrd_anchor_train25_qwen3_1_7b_nonthinking_seed20260811_transformers_v2/judge_remote_gpt56_luna_batch4
BRIDGE_MANIFEST=${ROOT}/dataset/train/anchor_bridge_qwen3_14b_h256_n8/manifest.jsonl
PYTHON=/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python

case "${MODE}" in
  teacher_from_anchor)
    MODEL=/storage/yukaichengLab/share/model/Qwen3-14B
    TRACE_BATCH_SIZE=2
    BASE_SEED=20260826
    INPUT_ARGS=(--anchor-root "${ANCHOR_ROOT}")
    ;;
  student_after_bridge)
    MODEL=/storage/yukaichengLab/share/model/Qwen3-1.7B
    TRACE_BATCH_SIZE=4
    BASE_SEED=20260826
    INPUT_ARGS=(--bridge-manifest "${BRIDGE_MANIFEST}")
    ;;
  student_from_anchor_prefix)
    MODEL=/storage/yukaichengLab/share/model/Qwen3-1.7B
    TRACE_BATCH_SIZE=4
    BASE_SEED=20260827
    INPUT_ARGS=(--anchor-root "${ANCHOR_ROOT}")
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    exit 3
    ;;
esac

CACHE_ROOT=/tmp/ocrd_full_rollout_${MODE}_${SLURM_JOB_ID}_${SHARD_ID}
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

echo "FULL_ROLLOUT_WORKER_START mode=${MODE} job=${SLURM_JOB_ID} shard=${SHARD_ID}/${NUM_SHARDS} node=${SLURMD_NODENAME} visible=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
"${PYTHON}" -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 1; free,total=torch.cuda.mem_get_info(); used=total-free; assert used < 1024*1024**2, f"visible GPU is not clean after CUDA context creation: used={used/2**20:.1f} MiB"; torch.manual_seed(20260826); x=torch.arange(8,device="cuda"); assert int(x.sum()) == 28; print("FULL_ROLLOUT_CUDA_WITNESS_OK",torch.cuda.get_device_name(),f"used_mib={used/2**20:.1f}")'

MAX_ARGS=()
if [[ -n "${MAX_RECORDS}" ]]; then
  MAX_ARGS=(--max-records "${MAX_RECORDS}")
fi

"${PYTHON}" "${MODULE}/run_full_rollout_batch.py" \
  --mode "${MODE}" \
  "${INPUT_ARGS[@]}" \
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
  --trace-batch-size "${TRACE_BATCH_SIZE}" \
  --gpu-memory-utilization 0.90

echo "FULL_ROLLOUT_WORKER_COMPLETE mode=${MODE} job=${SLURM_JOB_ID} shard=${SHARD_ID}/${NUM_SHARDS} node=${SLURMD_NODENAME}"
