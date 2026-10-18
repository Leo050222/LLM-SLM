#!/bin/bash
#SBATCH -J token_entropy_dist
#SBATCH -p yukaichenglab
#SBATCH --account=yukaicheng
#SBATCH --nodes=1
#SBATCH --nodelist=gnho020
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:H800:2
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/token_entropy_dist.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/token_entropy_dist.%j.err
set -Eeuo pipefail
ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
PY=/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python
SCRIPT=$ROOT/opd_scripts/token_entropy_distribution_20260914/distribution.py
OUT=$ROOT/opd_analysis/token_entropy_distribution_20260914
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
mkdir -p "$OUT"
IFS=',' read -ra DEVICES <<< "${CUDA_VISIBLE_DEVICES:?Slurm GPU binding required}"
test "${#DEVICES[@]}" -eq 2
CUDA_VISIBLE_DEVICES="${DEVICES[0]}" "$PY" -u "$SCRIPT" score --out "$OUT" --role student > "$OUT/student.log" 2>&1 &
S=$!
CUDA_VISIBLE_DEVICES="${DEVICES[1]}" "$PY" -u "$SCRIPT" score --out "$OUT" --role teacher > "$OUT/teacher.log" 2>&1 &
T=$!
trap 'kill "$S" "$T" 2>/dev/null || true' EXIT
wait "$S"
wait "$T"
"$PY" -u "$SCRIPT" summarize --out "$OUT"
"$PY" -u "$ROOT/opd_scripts/token_entropy_distribution_20260914/plot.py" --out "$OUT"
