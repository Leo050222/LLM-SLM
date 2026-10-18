#!/bin/bash
#SBATCH --job-name=dm_nt_gap
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:H800:1
#SBATCH --exclude=gnho018
#SBATCH --array=0-3%4
#SBATCH --time=04:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/dm_nt_gap.%A_%a.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/dm_nt_gap.%A_%a.err

set -euo pipefail
source /storage/yukaichengLab/wangxin/miniconda3/etc/profile.d/conda.sh
conda activate verl-opd

ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
RUN=${ROOT}/opd_eval_deepmath/ge6_test1000_native_nonthinking_seed20260811/qwen3_1_7b
TASK=${SLURM_ARRAY_TASK_ID}
ROLLOUTS=${RUN}/shards/shard_${TASK}/rollouts.jsonl
OUT=${RUN}/sampled_token_logprob_gap/shard_${TASK}
SCRIPT=${ROOT}/opd_scripts/run_deepmath_baseline_sample_gap.py

export PYTHONPATH=${ROOT}/verl
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton_dm_nt_gap_${SLURM_ARRAY_JOB_ID}_${TASK}
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_dm_nt_gap_${SLURM_ARRAY_JOB_ID}_${TASK}
mkdir -p "${OUT}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"

common=(
  --rollouts "${ROLLOUTS}"
  --output-dir "${OUT}"
  --student /storage/yukaichengLab/share/model/Qwen3-1.7B
  --teacher /storage/yukaichengLab/share/model/Qwen3-14B
)
python "${SCRIPT}" student "${common[@]}"
echo "NONTHINKING_STUDENT_LOGPROB_DONE=$(date --iso-8601=seconds)"
python "${SCRIPT}" teacher "${common[@]}"
echo "NONTHINKING_TEACHER_LOGPROB_DONE=$(date --iso-8601=seconds)"
