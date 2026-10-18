#!/bin/bash
#SBATCH --job-name=qwen_hard_nt_resume_4b
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:H800:1
#SBATCH --time=04:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/qwen_hard_nt_resume_4b.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/qwen_hard_nt_resume_4b.%j.err

set -euo pipefail
bash /storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/run_qwen3_hard_eval_nonthinking.sh qwen3_4b "" resume
