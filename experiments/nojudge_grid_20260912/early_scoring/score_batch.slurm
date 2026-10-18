#!/bin/bash
#SBATCH -J grid_early_score
#SBATCH -p yukaichenglab
#SBATCH --nodes=3
#SBATCH --ntasks=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=192G
#SBATCH --time=02:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/grid_early_score.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/grid_early_score.%j.err
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
PY=/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python
SCRIPT=/storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/nojudge_grid_early_score_20260912/score_batch.py
/soft/slurm/bin/srun --nodes=3 --ntasks=3 --ntasks-per-node=1 --cpus-per-task=64 --kill-on-bad-exit=1 "$PY" "$SCRIPT" "$SCORE_BATCH"
"$PY" "$SCRIPT" "$SCORE_BATCH" summarize
