#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH -J online_k8_score_parallel
set -euo pipefail
export OPD_CODE_ROOT=/storage/yukaichengLab/wangxin/test_eiloeo/opd_online_code/20260911_chunk64_k8_2nodes
export OPD_ONLINE_RUN=/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_deepmath/online_chunk64_k8_test1000_2nodes_seed20260911
export PYTHONPATH=/storage/yukaichengLab/wangxin/test_eiloeo/verl
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=''
test -f "$OPD_ONLINE_RUN/GENERATION_COMPLETE"
/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python "$OPD_ONLINE_RUN/evaluate_parallel.py" --run-dir "$OPD_ONLINE_RUN/results" --workers 16
touch "$OPD_ONLINE_RUN/EVALUATION_COMPLETE"
