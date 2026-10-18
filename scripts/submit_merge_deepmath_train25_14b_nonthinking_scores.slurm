#!/bin/bash
#SBATCH --job-name=dm14b_train25_nt_merge
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/dm14b_train25_nt_merge.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/dm14b_train25_nt_merge.%j.err

set -euo pipefail
source /storage/yukaichengLab/wangxin/miniconda3/etc/profile.d/conda.sh
conda activate verl-opd

ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
RUN=${ROOT}/opd_rollouts/deepmath_ge6_train25pct_qwen3_14b_native_nonthinking_seed20260811_single_node_8h800
SCORE_ROOT=${RUN}/scoring_rule_v1
SCRIPT=${ROOT}/opd_scripts/merge_deepmath_train25_nonthinking_scores.py

export PYTHONPATH=${ROOT}/opd_scripts:${ROOT}/verl
export TOKENIZERS_PARALLELISM=false
echo "DEEPMATH_TRAIN25_QWEN3_14B_NONTHINKING_SCORE_MERGE_START job=${SLURM_JOB_ID} node=${SLURMD_NODENAME} time=$(date --iso-8601=seconds)"
python "${SCRIPT}" --run-root "${RUN}" --score-root "${SCORE_ROOT}" \
  --heldout-same-protocol-accuracy 0.436
echo "DEEPMATH_TRAIN25_QWEN3_14B_NONTHINKING_SCORE_MERGE_OK job=${SLURM_JOB_ID} time=$(date --iso-8601=seconds)"
