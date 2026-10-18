#!/bin/bash
#SBATCH --job-name=q17b_math500_base
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --gres=gpu:H100:1
#SBATCH --time=12:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/q17b_math500_base.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/q17b_math500_base.%j.err

set -euo pipefail
bash /storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/run_qwen3_1_7b_math500_baseline.sh
