#!/bin/bash
#SBATCH --job-name=dm17_top16
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --nodelist=gnho020
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:H800:1
#SBATCH --array=0-3
#SBATCH --time=24:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/dm17_top16.%A_%a.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/dm17_top16.%A_%a.err

set -euo pipefail
source /storage/yukaichengLab/wangxin/miniconda3/etc/profile.d/conda.sh
conda activate verl-opd

ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
RUN=${ROOT}/opd_eval_deepmath/maxlen_seed20260809
SCRIPT=${ROOT}/opd_scripts/run_opd_logit_diagnostic.py
TASK=${SLURM_ARRAY_TASK_ID}
SOURCE=${RUN}/diagnostic_qwen3_1_7b/shard_${TASK}
OUT=${RUN}/diagnostic_qwen3_1_7b_top16/shard_${TASK}
DATA=${ROOT}/opd_data/deepmath_103k/sample_1000.shard${TASK}.parquet

export PYTHONPATH=${ROOT}/verl
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton_dm17_top16_${SLURM_ARRAY_JOB_ID}_${TASK}
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_dm17_top16_${SLURM_ARRAY_JOB_ID}_${TASK}
mkdir -p "${OUT}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"
cp "${SOURCE}/rollouts.jsonl" "${OUT}/rollouts.jsonl"
cp "${SOURCE}/run_config.json" "${OUT}/run_config.json"
ln -sfn "${SOURCE}/teacher_topk" "${OUT}/teacher_topk"

common=(
  --student /storage/yukaichengLab/share/model/Qwen3-1.7B
  --teacher /storage/yukaichengLab/share/model/Qwen3-14B
  --dataset-name deepmath_103k --data "${DATA}" --prompt-style deepmath --scorer math
  --score-timeout 20 --seed 20260809 --max-tokens 40512 --max-model-len 40960 --topk 16
)
python "${SCRIPT}" student --output-dir "${OUT}" "${common[@]}"
python "${SCRIPT}" summarize --output-dir "${OUT}" "${common[@]}"
echo "DEEPMATH_17B_TOP16_SHARD_COMPLETE=${TASK} $(date --iso-8601=seconds)"
