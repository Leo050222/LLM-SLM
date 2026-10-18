#!/bin/bash
#SBATCH --job-name=cuda_probe
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --nodelist=gnho020
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:H800:1
#SBATCH --time=00:05:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/cuda_probe.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/cuda_probe.%j.err

set -euo pipefail
source /storage/yukaichengLab/wangxin/miniconda3/etc/profile.d/conda.sh
conda activate verl-opd
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c 'import torch; print("CUDA_PROBE", torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_name(0)); x=torch.ones(1, device="cuda"); print("CUDA_TENSOR_OK", x.item())'
