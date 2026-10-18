#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH -J anchor14k_judge
#SBATCH -o /storage/yukaichengLab/wangxin/test_eiloeo/ocrd_anchor_train25_qwen3_1_7b_nonthinking_seed20260811_transformers_v2/judge_remote/logs/%j.out
#SBATCH -e /storage/yukaichengLab/wangxin/test_eiloeo/ocrd_anchor_train25_qwen3_1_7b_nonthinking_seed20260811_transformers_v2/judge_remote/logs/%j.err

set -euo pipefail

ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
RUN=${ROOT}/ocrd_anchor_train25_qwen3_1_7b_nonthinking_seed20260811_transformers_v2
MODULE=${ROOT}/ocrd_pipeline/anchor_selector
PYTHON=/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export SSL_CERT_FILE=$(${PYTHON} -c 'import certifi; print(certifi.where())')

# The login environment injects an unavailable proxy.  The OpenLux endpoint is
# directly reachable from the cluster, so force urllib to use the verified
# direct route.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

echo "ANCHOR14K_JUDGE_START job=${SLURM_JOB_ID} node=${SLURMD_NODENAME} workers=16"
${PYTHON} ${MODULE}/run_anchor_judge_batch.py \
  --input-root ${RUN} \
  --output-dir ${RUN}/judge_remote \
  --tokenizer /storage/yukaichengLab/share/model/Qwen3-1.7B \
  --config ${MODULE}/config.json \
  --secrets ${MODULE}/.secrets.env \
  --workers 16
echo "ANCHOR14K_JUDGE_COMPLETE job=${SLURM_JOB_ID} node=${SLURMD_NODENAME}"
