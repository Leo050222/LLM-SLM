#!/bin/bash
#SBATCH --job-name=weak_to_8b_prefix
#SBATCH --partition=yukaichenglab
#SBATCH --account=yukaicheng
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H800:8
#SBATCH --mem=512G
#SBATCH --time=2-00:00:00
#SBATCH --exclude=gnho005,gnho008,gnho009,gnho018
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/weak_to_8b_prefix.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/weak_to_8b_prefix.%j.err
set -Eeuo pipefail
export PYTHONDONTWRITEBYTECODE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
CODE=$ROOT/opd_scripts/weak_to_8b_prefix_handoff_20261005
OUT=$ROOT/opd_eval_prefix_handoff_weak_to_8b_20261005
PYTHON=/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python
mkdir -p "$OUT/logs"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
/soft/slurm/bin/srun --nodes=2 --ntasks=16 --ntasks-per-node=8 --cpus-per-task=8 --cpu-bind=cores --kill-on-bad-exit=1 --output="$OUT/logs/worker_%t.log" bash -lc 'export CUDA_VISIBLE_DEVICES=${SLURM_LOCALID}; export VLLM_CACHE_ROOT=/tmp/vllm_prefix_${SLURM_JOB_ID}_${SLURM_PROCID}; export TRITON_CACHE_DIR=/tmp/triton_prefix_${SLURM_JOB_ID}_${SLURM_PROCID}; exec /storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python /storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/weak_to_8b_prefix_handoff_20261005/generate_worker.py'
test "$(find "$OUT" -name 'worker_*_COMPLETE.json' | wc -l)" -eq 16
echo GENERATION_ALL_WORKERS_COMPLETE
