#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH --gres=gpu:6
#SBATCH -J opd_gkd_smoke
#SBATCH -o /storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/opd_gkd_smoke.%j.log
#SBATCH -e /storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/opd_gkd_smoke.%j.err
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G

set -euo pipefail

source /storage/yukaichengLab/wangxin/miniconda3/etc/profile.d/conda.sh
conda activate nli

export PYTHONPATH=/storage/yukaichengLab/wangxin/test_eiloeo/verl
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

mkdir -p /storage/yukaichengLab/wangxin/test_eiloeo/opd_logs
cd /storage/yukaichengLab/wangxin/test_eiloeo/verl

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start time: $(date)"
nvidia-smi

bash /storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/run_qwen3_4b_from_14b_gkd_smoke.sh

echo "End time: $(date)"
