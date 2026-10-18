#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH --gres=gpu:H800:1
#SBATCH --array=0-2%3
#SBATCH --exclude=gnho018
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH -J ocrd_anchor_nt
#SBATCH -o /storage/yukaichengLab/wangxin/test_eiloeo/ocrd_anchor_nonthinking_smoke_20260823/logs/%A_%a.out
#SBATCH -e /storage/yukaichengLab/wangxin/test_eiloeo/ocrd_anchor_nonthinking_smoke_20260823/logs/%A_%a.err

set -euo pipefail

ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
MODULE=${ROOT}/ocrd_pipeline/anchor_selector
TRACE=${ROOT}/opd_eval_deepmath/ge6_test1000_native_nonthinking_seed20260811/qwen3_1_7b/merged/rollouts.jsonl
RUN_ROOT=${ROOT}/ocrd_anchor_nonthinking_smoke_20260823
PYTHON=${ROOT}/../miniconda3/envs/verl-opd/bin/python
INDICES=(573 471 251)
DATASET_INDEX=${INDICES[$SLURM_ARRAY_TASK_ID]}
WORK_DIR=${RUN_ROOT}/trace_${DATASET_INDEX}
CACHE_DIR=${WORK_DIR}/cache

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export XDG_CACHE_HOME=${CACHE_DIR}/xdg
export TORCHINDUCTOR_CACHE_DIR=${CACHE_DIR}/torchinductor
export TRITON_CACHE_DIR=${CACHE_DIR}/triton

mkdir -p "${WORK_DIR}" "${XDG_CACHE_HOME}" "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"

echo "ANCHOR_NONTHINKING_SMOKE_START job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} dataset_index=${DATASET_INDEX} node=${SLURMD_NODENAME}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader

"${PYTHON}" -c 'import torch; torch.manual_seed(20260823); x=torch.randn(256,256,device="cuda"); y=x@x; torch.cuda.synchronize(); print("CUDA_WITNESS_OK", float(y.sum()))'

"${PYTHON}" "${MODULE}/run_anchor_selector.py" score-student \
  --trace "${TRACE}" \
  --dataset-index "${DATASET_INDEX}" \
  --work-dir "${WORK_DIR}"

"${PYTHON}" "${MODULE}/run_anchor_selector.py" score-teacher \
  --trace "${TRACE}" \
  --dataset-index "${DATASET_INDEX}" \
  --work-dir "${WORK_DIR}"

"${PYTHON}" "${MODULE}/run_anchor_selector.py" candidates \
  --trace "${TRACE}" \
  --dataset-index "${DATASET_INDEX}" \
  --work-dir "${WORK_DIR}"

touch "${WORK_DIR}/GPU_SCORING_COMPLETE"
echo "ANCHOR_NONTHINKING_SMOKE_GPU_COMPLETE dataset_index=${DATASET_INDEX}"
