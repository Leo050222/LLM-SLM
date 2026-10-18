#!/bin/bash
#SBATCH --job-name=dm_base_gap
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:H800:1
#SBATCH --array=0-3
#SBATCH --time=04:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/dm_base_gap.%A_%a.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/dm_base_gap.%A_%a.err

set -euo pipefail
source /storage/yukaichengLab/wangxin/miniconda3/etc/profile.d/conda.sh
conda activate verl-opd

ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
RUN=${ROOT}/opd_eval_deepmath/ge6_test1000_qwen3_1_7b_baseline_seed20260811
TASK=${SLURM_ARRAY_TASK_ID}
ROLLOUTS=${RUN}/shards/shard_${TASK}/rollouts.jsonl
OUT=${RUN}/sampled_token_logprob_gap/shard_${TASK}
SCRIPT=${ROOT}/opd_scripts/run_deepmath_baseline_sample_gap.py

export PYTHONPATH=${ROOT}/verl
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton_dm_base_gap_${SLURM_ARRAY_JOB_ID}_${TASK}
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_dm_base_gap_${SLURM_ARRAY_JOB_ID}_${TASK}
mkdir -p "${OUT}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"

common=(
  --rollouts "${ROLLOUTS}"
  --output-dir "${OUT}"
  --student /storage/yukaichengLab/share/model/Qwen3-1.7B
  --teacher /storage/yukaichengLab/share/model/Qwen3-14B
)
python "${SCRIPT}" student "${common[@]}"
echo "BASELINE_STUDENT_LOGPROB_DONE=$(date --iso-8601=seconds)"
python "${SCRIPT}" teacher "${common[@]}"
echo "BASELINE_TEACHER_LOGPROB_DONE=$(date --iso-8601=seconds)"
