#!/bin/bash
#SBATCH --job-name=dm_s16_trace
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --nodelist=gnho016
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:H800:1
#SBATCH --array=0-3
#SBATCH --time=24:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/dm_s16_trace.%A_%a.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/dm_s16_trace.%A_%a.err

set -euo pipefail
source /storage/yukaichengLab/wangxin/miniconda3/etc/profile.d/conda.sh
conda activate verl-opd

ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
RUN=${ROOT}/opd_eval_deepmath/maxlen_seed20260809
TASK=${SLURM_ARRAY_TASK_ID}
SCRIPT=${ROOT}/opd_scripts/run_deepmath_student_top16_trace_diagnostic.py
ROLLOUTS=${RUN}/diagnostic_qwen3_1_7b/shard_${TASK}/rollouts.jsonl
TEACHER_ROLLOUTS=${RUN}/merged/qwen3_14b_rollouts.jsonl
OUT=${RUN}/student_top16_trace_diagnostic/shard_${TASK}

export PYTHONPATH=${ROOT}/verl
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton_dm_s16_trace_${SLURM_ARRAY_JOB_ID}_${TASK}
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_dm_s16_trace_${SLURM_ARRAY_JOB_ID}_${TASK}
mkdir -p "${OUT}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"

common=(
  --rollouts "${ROLLOUTS}"
  --teacher-rollouts "${TEACHER_ROLLOUTS}"
  --output-dir "${OUT}"
  --student /storage/yukaichengLab/share/model/Qwen3-1.7B
  --teacher /storage/yukaichengLab/share/model/Qwen3-14B
)
python "${SCRIPT}" student "${common[@]}"
echo "STUDENT_TOP16_DONE=$(date --iso-8601=seconds)"
python "${SCRIPT}" teacher "${common[@]}"
echo "TEACHER_ON_STUDENT_TOP16_DONE=$(date --iso-8601=seconds)"
python "${SCRIPT}" summarize "${common[@]}"
echo "STUDENT_TOP16_TRACE_SHARD_COMPLETE=${TASK} $(date --iso-8601=seconds)"
