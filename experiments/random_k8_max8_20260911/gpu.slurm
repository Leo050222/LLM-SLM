#!/bin/bash
#SBATCH -J random_k8_max8
#SBATCH -p yukaichenglab
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:H800:8
#SBATCH --exclude=gnho009
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/random_k8_max8.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/random_k8_max8.%j.err
set -Eeuo pipefail
export OPD_CODE_ROOT=/storage/yukaichengLab/wangxin/test_eiloeo/opd_online_code/20260911_chunk64_k8_2nodes
export OPD_ONLINE_RUN=/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_hard/random_k8_max8_20260911
export OPD_PYTHON=/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python
/soft/slurm/bin/srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --cpus-per-task=64 --cpu-bind=cores --kill-on-bad-exit=1 bash /storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/random_k8_max8_20260911/node.sh
