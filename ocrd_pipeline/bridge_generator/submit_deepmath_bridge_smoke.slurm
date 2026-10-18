#!/usr/bin/env bash
#SBATCH --job-name=ocrd_bridge
#SBATCH --partition=yukaichenglab
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:H800:1
#SBATCH --time=01:00:00
#SBATCH --exclude=gnho018
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/ocrd_anchor_smoke_20260822/bridge_logs/%A_%a.out
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/ocrd_anchor_smoke_20260822/bridge_logs/%A_%a.err

set -euo pipefail

PYTHON=/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python
MODULE=/storage/yukaichengLab/wangxin/test_eiloeo/ocrd_pipeline/bridge_generator
RUN_ROOT=/storage/yukaichengLab/wangxin/test_eiloeo/ocrd_anchor_smoke_20260822
INDICES=(338 216 485)
DATASET_INDEX=${INDICES[$SLURM_ARRAY_TASK_ID]}
BRIDGE_H=${BRIDGE_H:-32}
TRACE_DIR=${RUN_ROOT}/trace_${DATASET_INDEX}
CACHE_DIR=${TRACE_DIR}/bridge_cache_h${BRIDGE_H}

export XDG_CACHE_HOME=${CACHE_DIR}/xdg
export TORCHINDUCTOR_CACHE_DIR=${CACHE_DIR}/torchinductor
export TRITON_CACHE_DIR=${CACHE_DIR}/triton
export TOKENIZERS_PARALLELISM=false
mkdir -p "${RUN_ROOT}/bridge_logs" "${XDG_CACHE_HOME}" \
  "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"${PYTHON}" -c 'import torch; torch.manual_seed(20260822); x=torch.randn(256,256,device="cuda"); print("BRIDGE_CUDA_WITNESS",torch.cuda.get_device_name(),float((x@x).sum()))'

"${PYTHON}" "${MODULE}/run_bridge_generator.py" \
  --anchor "${TRACE_DIR}/selected_anchor.json" \
  --output "${TRACE_DIR}/bridge_h${BRIDGE_H}.json" \
  --h "${BRIDGE_H}" \
  --temperature 1.0 \
  --top-p 1.0 \
  --top-k 0 \
  --seed "$((20260822 + DATASET_INDEX))"
