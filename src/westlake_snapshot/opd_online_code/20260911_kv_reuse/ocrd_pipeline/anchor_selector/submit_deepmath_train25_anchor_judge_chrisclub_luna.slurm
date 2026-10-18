#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH -J anchor14k_luna
#SBATCH -o /storage/yukaichengLab/wangxin/test_eiloeo/ocrd_anchor_train25_qwen3_1_7b_nonthinking_seed20260811_transformers_v2/judge_remote_gpt56_luna_batch4/logs/%j.out
#SBATCH -e /storage/yukaichengLab/wangxin/test_eiloeo/ocrd_anchor_train25_qwen3_1_7b_nonthinking_seed20260811_transformers_v2/judge_remote_gpt56_luna_batch4/logs/%j.err

set -euo pipefail

ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
RUN=${ROOT}/ocrd_anchor_train25_qwen3_1_7b_nonthinking_seed20260811_transformers_v2
MODULE=${ROOT}/ocrd_pipeline/anchor_selector
OUTPUT=${RUN}/judge_remote_gpt56_luna_batch4
PYTHON=/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

# The Chris Club endpoint is directly reachable from compute nodes.  Avoid the
# login environment's proxy, which is not reachable inside Slurm allocations.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

mkdir -p ${OUTPUT}/logs
echo "ANCHOR14K_LUNA_START job=${SLURM_JOB_ID} node=${SLURMD_NODENAME} workers=64 candidate_batch_size=4"
${PYTHON} ${MODULE}/run_anchor_judge_batch.py \
  --input-root ${RUN} \
  --output-dir ${OUTPUT} \
  --tokenizer /storage/yukaichengLab/share/model/Qwen3-1.7B \
  --config ${MODULE}/config_gpt56_luna.json \
  --secrets ${MODULE}/.secrets.env \
  --workers 64 \
  --candidate-batch-size 4
echo "ANCHOR14K_LUNA_COMPLETE job=${SLURM_JOB_ID} node=${SLURMD_NODENAME}"
