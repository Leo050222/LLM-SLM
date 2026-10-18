#!/bin/bash
#SBATCH --job-name=student64_add56
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=2
#SBATCH --nodelist=gnho033,gnho034
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:H800:8
#SBATCH --mem=512G
#SBATCH --time=12:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/s64.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/s64.%j.err
set -Eeuo pipefail
source /storage/yukaichengLab/wangxin/miniconda3/etc/profile.d/conda.sh
conda activate verl-opd
export ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
export OUT=$ROOT/opd_eval_deepmath/student_only64_20260910
export SCRIPT=$ROOT/opd_scripts/student_only64_20260910/run.py
export PYTHONPATH=$ROOT/opd_scripts:$ROOT/verl
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

trap 'rc=$?; echo "BATCH_EXIT rc=$rc time=$(date -Is)" >&2' EXIT
python "$SCRIPT" prepare --out "$OUT"
/soft/slurm/bin/srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --cpus-per-task=64 --cpu-bind=cores --kill-on-bad-exit=1 bash "$ROOT/opd_scripts/student_only64_20260910/node.sh"
python "$SCRIPT" verify --out "$OUT"
