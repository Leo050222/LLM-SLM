#!/usr/bin/env bash
#SBATCH --job-name=bridge256_smoke
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:H800:1
#SBATCH --exclude=gnho009,gnho018
#SBATCH --time=01:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/ocrd_anchor_train25_qwen3_1_7b_nonthinking_seed20260811_transformers_v2/bridge_qwen3_14b_h256_n8_smoke32/logs/%j.out
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/ocrd_anchor_train25_qwen3_1_7b_nonthinking_seed20260811_transformers_v2/bridge_qwen3_14b_h256_n8_smoke32/logs/%j.err

set -euo pipefail

ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
MODULE=${ROOT}/ocrd_pipeline/bridge_generator
ANCHOR_ROOT=${ROOT}/ocrd_anchor_train25_qwen3_1_7b_nonthinking_seed20260811_transformers_v2/judge_remote_gpt56_luna_batch4
OUTPUT_ROOT=${ROOT}/ocrd_anchor_train25_qwen3_1_7b_nonthinking_seed20260811_transformers_v2/bridge_qwen3_14b_h256_n8_smoke32
PYTHON=/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python
MODEL=/storage/yukaichengLab/share/model/Qwen3-14B
CACHE_ROOT=/tmp/ocrd_bridge_h256_smoke_${SLURM_JOB_ID}

export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=8 VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export XDG_CACHE_HOME=${CACHE_ROOT}/xdg
export VLLM_CACHE_ROOT=${CACHE_ROOT}/vllm
export TORCHINDUCTOR_CACHE_DIR=${CACHE_ROOT}/torchinductor
export TRITON_CACHE_DIR=${CACHE_ROOT}/triton
mkdir -p "${OUTPUT_ROOT}/logs" "${XDG_CACHE_HOME}" "${VLLM_CACHE_ROOT}" \
  "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"

echo "BRIDGE_SMOKE_START job=${SLURM_JOB_ID} node=${SLURMD_NODENAME}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
"${PYTHON}" -c 'import torch; assert torch.cuda.is_available(); x=torch.arange(8,device="cuda"); assert int(x.sum()) == 28; print("BRIDGE_CUDA_WITNESS_OK",torch.cuda.get_device_name())'

"${PYTHON}" "${MODULE}/run_bridge_batch.py" \
  --anchor-root "${ANCHOR_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --shard-id 0 --num-shards 1 --max-records 32 \
  --h 256 --teacher "${MODEL}" \
  --temperature 1.0 --top-p 1.0 --top-k 0 --seed 20260822 \
  --first-step-logprobs 16 --max-model-len 4096 \
  --num-rollouts 8 --trace-batch-size 8 --gpu-memory-utilization 0.90

"${PYTHON}" "${MODULE}/verify_bridge_batch.py" \
  --anchor-root "${ANCHOR_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --num-shards 1 --max-records 32 --expected-count 32 --num-rollouts 8 \
  --h 256 --teacher "${MODEL}"
echo "BRIDGE_SMOKE_COMPLETE job=${SLURM_JOB_ID} node=${SLURMD_NODENAME}"
