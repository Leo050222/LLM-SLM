#!/bin/bash
#SBATCH --job-name=teacher_prefix_handoff
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --gres=gpu:H800:1
#SBATCH --time=08:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/teacher_prefix_handoff.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/teacher_prefix_handoff.%j.err
set -euo pipefail
source /storage/yukaichengLab/wangxin/miniconda3/etc/profile.d/conda.sh
conda activate verl-opd
ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
export PYTHONPATH=${ROOT}/opd_scripts:${ROOT}/verl
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export VLLM_CACHE_ROOT=/tmp/vllm_prefix_${SLURM_JOB_ID}
export TRITON_CACHE_DIR=/tmp/triton_prefix_${SLURM_JOB_ID}
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
python "${ROOT}/opd_scripts/teacher_prefix_handoff_20260909/run.py" --output "${ROOT}/opd_eval_deepmath/teacher_prefix_handoff_20260909" "$@"
