#!/bin/bash
#SBATCH --job-name=deepmath17_eval
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:H100:1
#SBATCH --array=0-3
#SBATCH --time=08:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/deepmath17_eval.%A_%a.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/deepmath17_eval.%A_%a.err

set -euo pipefail

source /storage/yukaichengLab/wangxin/miniconda3/etc/profile.d/conda.sh
conda activate verl-opd

ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
SCRIPT=${ROOT}/opd_scripts/run_opd_logit_diagnostic.py
DATA_ROOT=${ROOT}/opd_data/deepmath_103k
OUTPUT=${ROOT}/opd_eval_deepmath/full_seed20260809
STUDENT=/storage/yukaichengLab/share/model/Qwen3-1.7B
TEACHER=/storage/yukaichengLab/share/model/Qwen3-14B
TASK=${SLURM_ARRAY_TASK_ID}

export PYTHONPATH=${ROOT}/verl
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=8
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton_deepmath17_${SLURM_ARRAY_JOB_ID}_${TASK}
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_deepmath17_${SLURM_ARRAY_JOB_ID}_${TASK}

data=${DATA_ROOT}/sample_1000.shard${TASK}.parquet
out=${OUTPUT}/diagnostic_qwen3_1_7b/shard_${TASK}
mkdir -p "${out}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"

echo "ARRAY_JOB=${SLURM_ARRAY_JOB_ID} TASK=${TASK} NODE=${SLURMD_NODENAME} START=$(date --iso-8601=seconds)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

common=(
  --student "${STUDENT}"
  --teacher "${TEACHER}"
  --dataset-name deepmath_103k
  --data "${data}"
  --prompt-style deepmath
  --scorer math
  --score-timeout 20
  --seed 20260809
  --max-tokens 4096
  --max-model-len 8192
  --topk 128
)

python "${SCRIPT}" generate --output-dir "${out}" "${common[@]}"
echo "PHASE_GENERATE_DONE=$(date --iso-8601=seconds)"
python "${SCRIPT}" teacher --output-dir "${out}" "${common[@]}"
echo "PHASE_TEACHER_DONE=$(date --iso-8601=seconds)"
python "${SCRIPT}" student --output-dir "${out}" "${common[@]}"
echo "PHASE_STUDENT_DONE=$(date --iso-8601=seconds)"
python "${SCRIPT}" summarize --output-dir "${out}" "${common[@]}"
echo "DIAGNOSTIC_17B_SHARD_COMPLETE=${TASK} $(date --iso-8601=seconds)"
