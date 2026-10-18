#!/bin/bash
#SBATCH --job-name=student64_score
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=3
#SBATCH --ntasks=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=512G
#SBATCH --time=12:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/s64_score.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/s64_score.%j.err
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

export CUDA_VISIBLE_DEVICES=""
trap 'rc=$?; echo "SCORE_BATCH_EXIT rc=$rc time=$(date -Is)" >&2' EXIT
/soft/slurm/bin/srun --nodes=3 --ntasks=3 --ntasks-per-node=1 --cpus-per-task=96 --cpu-bind=cores --kill-on-bad-exit=1 --output="$ROOT/opd_logs/s64_score.${SLURM_JOB_ID}.rank%t.log" --error="$ROOT/opd_logs/s64_score.${SLURM_JOB_ID}.rank%t.err" bash -c 'python "$SCRIPT" score --out "$OUT" --rank "$SLURM_PROCID" --workers 80'
python "$SCRIPT" merge --out "$OUT"
