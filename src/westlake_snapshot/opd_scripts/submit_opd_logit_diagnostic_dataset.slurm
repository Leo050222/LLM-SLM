#!/bin/bash
#SBATCH --job-name=opd_logits
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:H800:1
#SBATCH --time=12:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/opd_logits_%x.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/opd_logits_%x.%j.err

set -euo pipefail

source /storage/yukaichengLab/wangxin/miniconda3/etc/profile.d/conda.sh
conda activate verl-opd

ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
SCRIPT=${ROOT}/opd_scripts/run_opd_logit_diagnostic.py
DATASET=${OPD_DIAG_DATASET:?Set OPD_DIAG_DATASET to aime24, aime25, or hmmt25}
RUN_KIND=${OPD_DIAG_RUN:-full}

case "${DATASET}" in
  aime24)
    DATA=${ROOT}/opd_eval_hard/modelscope_cache/evalscope/aime24/test-00000-of-00001.parquet
    ;;
  aime25)
    DATA=${ROOT}/opd_eval_hard/modelscope_cache/evalscope/aime25/test.jsonl
    ;;
  hmmt25)
    DATA=${ROOT}/opd_eval_hard/modelscope_cache/evalscope/hmmt_feb_2025/data/train-00000-of-00001.parquet
    ;;
  *)
    echo "Unsupported OPD_DIAG_DATASET=${DATASET}" >&2
    exit 2
    ;;
esac

OUTPUT=${ROOT}/opd_logit_diagnostics/${DATASET}_step0_t1_${RUN_KIND}_seed20260806

export PYTHONPATH=${ROOT}/verl
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=8
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton_opd_logits_${SLURM_JOB_ID}
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_opd_logits_${SLURM_JOB_ID}

mkdir -p "${OUTPUT}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${ROOT}/opd_logs"

echo "JOB_ID=${SLURM_JOB_ID}"
echo "NODE=${SLURMD_NODENAME}"
echo "DATASET=${DATASET}"
echo "DATA=${DATA}"
echo "RUN_KIND=${RUN_KIND}"
echo "OUTPUT=${OUTPUT}"
echo "START=$(date --iso-8601=seconds)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

LIMIT_ARGS=()
if [[ -n "${OPD_DIAG_LIMIT:-}" ]]; then
  LIMIT_ARGS=(--limit "${OPD_DIAG_LIMIT}")
fi

python "${SCRIPT}" generate --dataset-name "${DATASET}" --data "${DATA}" --output-dir "${OUTPUT}" "${LIMIT_ARGS[@]}"
echo "PHASE_GENERATE_DONE=$(date --iso-8601=seconds)"
python "${SCRIPT}" teacher --output-dir "${OUTPUT}"
echo "PHASE_TEACHER_DONE=$(date --iso-8601=seconds)"
python "${SCRIPT}" student --output-dir "${OUTPUT}"
echo "PHASE_STUDENT_DONE=$(date --iso-8601=seconds)"
python "${SCRIPT}" summarize --output-dir "${OUTPUT}"
echo "DIAGNOSTIC_COMPLETE=$(date --iso-8601=seconds)"
