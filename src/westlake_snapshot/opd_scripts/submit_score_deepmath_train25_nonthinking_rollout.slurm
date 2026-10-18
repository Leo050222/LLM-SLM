#!/bin/bash
#SBATCH --job-name=dm_train25_nt_score
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --array=0-15%16
#SBATCH --time=02:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/dm_train25_nt_score.%A_%a.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/dm_train25_nt_score.%A_%a.err

set -euo pipefail
source /storage/yukaichengLab/wangxin/miniconda3/etc/profile.d/conda.sh
conda activate verl-opd

ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
RUN=${ROOT}/opd_rollouts/deepmath_ge6_train25pct_qwen3_1_7b_native_nonthinking_seed20260811
SCORE_ROOT=${RUN}/scoring_rule_v1
SCRIPT=${ROOT}/opd_scripts/score_deepmath_train25_nonthinking_rollout.py
SHARD=$(printf '%02d' "${SLURM_ARRAY_TASK_ID}")
ROLLOUTS=${RUN}/shards/shard_${SHARD}/rollouts.jsonl
FINAL=${SCORE_ROOT}/shards/shard_${SHARD}
ATTEMPT=${SCORE_ROOT}/attempts/job_${SLURM_ARRAY_JOB_ID}_task_${SHARD}

if [[ -e "${FINAL}" || -e "${ATTEMPT}" ]]; then
  echo "REFUSE_OVERWRITE final=${FINAL} attempt=${ATTEMPT}" >&2
  exit 73
fi
if [[ ! -f "${ROLLOUTS}" ]]; then
  echo "ROLLOUTS_MISSING path=${ROLLOUTS}" >&2
  exit 66
fi
mkdir -p "${SCORE_ROOT}/attempts" "${SCORE_ROOT}/shards"

export PYTHONPATH=${ROOT}/opd_scripts:${ROOT}/verl
export TOKENIZERS_PARALLELISM=false
echo "DEEPMATH_TRAIN25_NATIVE_NONTHINKING_SCORE_START job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} node=${SLURMD_NODENAME} time=$(date --iso-8601=seconds)"
python "${SCRIPT}" \
  --rollouts "${ROLLOUTS}" \
  --output-dir "${ATTEMPT}" \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --num-shards 16 \
  --expected-rows 888 \
  --score-timeout 20
mv "${ATTEMPT}" "${FINAL}"
echo "DEEPMATH_TRAIN25_NATIVE_NONTHINKING_SCORE_OK job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} final=${FINAL} time=$(date --iso-8601=seconds)"
