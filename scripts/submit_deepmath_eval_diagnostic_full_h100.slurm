#!/bin/bash
#SBATCH --job-name=deepmath_eval
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:H100:1
#SBATCH --array=0-4
#SBATCH --time=08:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/deepmath_eval.%A_%a.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/deepmath_eval.%A_%a.err

set -euo pipefail

source /storage/yukaichengLab/wangxin/miniconda3/etc/profile.d/conda.sh
conda activate verl-opd

ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
SCRIPT=${ROOT}/opd_scripts/run_opd_logit_diagnostic.py
DATA_ROOT=${ROOT}/opd_data/deepmath_103k
OUTPUT=${ROOT}/opd_eval_deepmath/full_seed20260809
TASK=${SLURM_ARRAY_TASK_ID}

export PYTHONPATH=${ROOT}/verl
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=8
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton_deepmath_${SLURM_ARRAY_JOB_ID}_${TASK}
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_deepmath_${SLURM_ARRAY_JOB_ID}_${TASK}

mkdir -p "${OUTPUT}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"
echo "ARRAY_JOB=${SLURM_ARRAY_JOB_ID} TASK=${TASK} NODE=${SLURMD_NODENAME} START=$(date --iso-8601=seconds)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

common=(
  --dataset-name deepmath_103k
  --prompt-style deepmath
  --scorer math
  --score-timeout 20
  --seed 20260809
  --max-tokens 4096
  --max-model-len 8192
  --topk 128
)

if [[ "${TASK}" -lt 4 ]]; then
  data=${DATA_ROOT}/sample_1000.shard${TASK}.parquet
  out=${OUTPUT}/diagnostic/shard_${TASK}
  mkdir -p "${out}"
  python "${SCRIPT}" generate --data "${data}" --output-dir "${out}" "${common[@]}"
  echo "PHASE_GENERATE_DONE=$(date --iso-8601=seconds)"
  python "${SCRIPT}" teacher --output-dir "${out}" --topk 128 --seed 20260809 --max-tokens 4096 --max-model-len 8192
  echo "PHASE_TEACHER_DONE=$(date --iso-8601=seconds)"
  python "${SCRIPT}" student --output-dir "${out}" --topk 128 --seed 20260809 --max-tokens 4096 --max-model-len 8192
  echo "PHASE_STUDENT_DONE=$(date --iso-8601=seconds)"
  python "${SCRIPT}" summarize --output-dir "${out}" --topk 128 --seed 20260809 --max-tokens 4096 --max-model-len 8192
  echo "DIAGNOSTIC_SHARD_COMPLETE=${TASK} $(date --iso-8601=seconds)"
else
  out=${OUTPUT}/qwen3_14b
  mkdir -p "${out}"
  python "${SCRIPT}" generate \
    --data "${DATA_ROOT}/sample_1000.parquet" \
    --output-dir "${out}" \
    --student /storage/yukaichengLab/share/model/Qwen3-14B \
    --teacher /storage/yukaichengLab/share/model/Qwen3-14B \
    "${common[@]}"
  echo "QWEN3_14B_EVAL_COMPLETE=$(date --iso-8601=seconds)"
fi
