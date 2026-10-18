#!/bin/bash
#SBATCH -J qwen3_8b_greedy
#SBATCH -p yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:H800:8
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/qwen3_8b_greedy_hard_three.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/qwen3_8b_greedy_hard_three.%j.err
set -Eeuo pipefail
bash /storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/qwen3_greedy_hard_three_20261005/run_job.sh qwen3_8b
