#!/bin/bash
#SBATCH --job-name=opd_base_eval
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --gres=gpu:H800:2
#SBATCH --time=01:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/opd_base_eval.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/opd_base_eval.%j.err

set -euo pipefail

source /storage/yukaichengLab/wangxin/miniconda3/etc/profile.d/conda.sh
conda activate verl-opd

export PYTHONPATH=/storage/yukaichengLab/wangxin/test_eiloeo/verl
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=8

SCRIPT=/storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/eval_qwen_opd_models.py
DATA=/storage/yukaichengLab/wangxin/test_eiloeo/opd_data/gsm8k_math/validation_512.parquet
OUT=/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval/qwen3_gsm8k_math_512

mkdir -p "$OUT"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start time: $(date --iso-8601=seconds)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" \
  --model /storage/yukaichengLab/share/model/Qwen3-4B \
  --model-name qwen3_4b_base \
  --data "$DATA" \
  --output "$OUT/qwen3_4b_base.jsonl" \
  --tp 1

CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" \
  --model /storage/yukaichengLab/wangxin/test_eiloeo/opd_checkpoints/qwen3_4b_from_14b_gkd_gsm8k_math/global_step_234/actor/huggingface \
  --model-name qwen3_4b_opd_step234 \
  --data "$DATA" \
  --output "$OUT/qwen3_4b_opd_step234.jsonl" \
  --tp 1

CUDA_VISIBLE_DEVICES=0,1 python "$SCRIPT" \
  --model /storage/yukaichengLab/share/model/Qwen3-14B \
  --model-name qwen3_14b_base \
  --data "$DATA" \
  --output "$OUT/qwen3_14b_base.jsonl" \
  --tp 2

echo "End time: $(date --iso-8601=seconds)"
