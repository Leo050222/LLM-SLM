#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH --nodes=1
#SBATCH --nodelist=nv-h100-029
#SBATCH --ntasks=4
#SBATCH --gres=gpu:H100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --time=01:00:00
#SBATCH -J kv_reuse_validate
set -euo pipefail
export OPD_CODE_ROOT=/storage/yukaichengLab/wangxin/test_eiloeo/opd_online_code/20260911_kv_reuse_v2
export OPD_KV_RUN=/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_deepmath/kv_reuse_validation_20260911_v2
export OPD_PYTHON=/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python
/soft/slurm/bin/srun --kill-on-bad-exit=1 --gpus-per-task=H100:1 --gpu-bind=single:1 --output="$OPD_KV_RUN/logs/%j_rank%t.log" bash "$OPD_KV_RUN/rank.sh"
