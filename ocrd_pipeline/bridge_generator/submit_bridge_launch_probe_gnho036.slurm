#!/usr/bin/env bash
#SBATCH --job-name=bridge_launch_probe
#SBATCH --partition=yukaichenglab
#SBATCH --nodelist=gnho036
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/ocrd_anchor_train25_qwen3_1_7b_nonthinking_seed20260811_transformers_v2/bridge_qwen3_14b_h256_n8_seed20260822/logs/launch_probe_%j.out
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/ocrd_anchor_train25_qwen3_1_7b_nonthinking_seed20260811_transformers_v2/bridge_qwen3_14b_h256_n8_seed20260822/logs/launch_probe_%j.err

set -euo pipefail

ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
MODULE=${ROOT}/ocrd_pipeline/bridge_generator
OUTPUT_ROOT=${ROOT}/ocrd_anchor_train25_qwen3_1_7b_nonthinking_seed20260811_transformers_v2/bridge_qwen3_14b_h256_n8_seed20260822
PROBE=${OUTPUT_ROOT}/.launch_probe_${SLURM_JOB_ID}

echo "BRIDGE_LAUNCH_PROBE_START job=${SLURM_JOB_ID} node=${SLURMD_NODENAME}"
test -r "${MODULE}/run_bridge_batch.py"
test -r "${MODULE}/run_bridge_h256_shard_worker.sh"
dd if=/dev/zero of="${PROBE}" bs=1M count=1 status=none
test "$(stat -c %s "${PROBE}")" -eq 1048576
unlink "${PROBE}"
echo "BRIDGE_LAUNCH_PROBE_COMPLETE job=${SLURM_JOB_ID} node=${SLURMD_NODENAME}"
