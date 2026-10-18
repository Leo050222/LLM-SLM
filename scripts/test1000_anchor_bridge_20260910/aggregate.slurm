#!/bin/bash
#SBATCH --job-name=t1k_sbridge_merge
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_deepmath/ge6_test1000_anchor_bridge_h256_n8x8_seed20260910/logs/aggregate_%j.out
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_deepmath/ge6_test1000_anchor_bridge_h256_n8x8_seed20260910/logs/aggregate_%j.err

set -euo pipefail
ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
EXP=${ROOT}/opd_eval_deepmath/ge6_test1000_anchor_bridge_h256_n8x8_seed20260910
PYTHON=/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python
SCRIPT=${ROOT}/opd_scripts/test1000_anchor_bridge_20260910/aggregate_n8x8.py
"${PYTHON}" "${SCRIPT}" --experiment-root "${EXP}" --output-dir "${EXP}/analysis_n8x8" --replicates 8
