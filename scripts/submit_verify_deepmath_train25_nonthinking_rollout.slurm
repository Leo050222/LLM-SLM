#!/bin/bash
#SBATCH --job-name=dm_train25_nt_verify
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/dm_train25_nt_verify.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/dm_train25_nt_verify.%j.err

set -euo pipefail
source /storage/yukaichengLab/wangxin/miniconda3/etc/profile.d/conda.sh
conda activate verl-opd

ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
SOURCE=${ROOT}/opd_data/deepmath_ge6_opd_train25pct_perdifficulty_seed20260823/train.parquet
OUTPUT=${ROOT}/opd_rollouts/deepmath_ge6_train25pct_qwen3_1_7b_native_nonthinking_seed20260811
SCRIPT=${ROOT}/opd_scripts/verify_deepmath_train25_nonthinking_rollout.py

export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
echo "DEEPMATH_TRAIN25_NATIVE_NONTHINKING_VERIFY_START job=${SLURM_JOB_ID} node=${SLURMD_NODENAME} time=$(date --iso-8601=seconds)"
python "${SCRIPT}" --source "${SOURCE}" --output-root "${OUTPUT}"
echo "DEEPMATH_TRAIN25_NATIVE_NONTHINKING_VERIFY_OK job=${SLURM_JOB_ID} time=$(date --iso-8601=seconds)"
