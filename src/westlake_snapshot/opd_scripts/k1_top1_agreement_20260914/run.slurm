#!/bin/bash
#SBATCH -J k1_top1_agree
#SBATCH -p yukaichenglab
#SBATCH --account=yukaicheng
#SBATCH --nodes=1
#SBATCH --nodelist=gnho038
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:H800:2
#SBATCH --mem=160G
#SBATCH --time=01:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/k1_top1_agree.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/k1_top1_agree.%j.err
set -Eeuo pipefail
ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
PY=/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python
SCRIPT=$ROOT/opd_scripts/k1_top1_agreement_20260914/probe.py
OUT=$ROOT/opd_analysis/k1_top1_agreement_20260914
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
