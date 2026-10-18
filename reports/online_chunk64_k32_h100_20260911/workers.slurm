#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH --nodes=1
#SBATCH --nodelist=nv-h100-007
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:H100:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=960G
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH -J online_k32_h100
set -euo pipefail
export PATH=/soft/slurm/bin:$PATH
: "${OPD_CODE_ROOT:?}"
: "${OPD_ONLINE_RUN:?}"
: "${OPD_PYTHON:?}"
case "$OPD_ONLINE_RUN" in
 /storage/yukaichengLab/wangxin/test_eiloeo/*) ;;
 *) exit 2 ;;
esac
mkdir -p "$OPD_ONLINE_RUN/logs"
# One srun across both complete nodes; rank-to-GPU binding is owned by Slurm.
srun --kill-on-bad-exit=1 --nodes=1 --ntasks=8 --ntasks-per-node=8 \
 --gpus-per-task=H100:1 --gpu-bind=single:1 --cpus-per-task=16 \
 --output="$OPD_ONLINE_RUN/logs/%j_rank%t.log" \
 bash "$OPD_CODE_ROOT/ocrd_pipeline/online_rollout/worker_rank.sh"
