#!/bin/bash
#SBATCH --job-name=t1k_anchor_luna
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_deepmath/ge6_test1000_anchor_bridge_h256_n8x8_seed20260910/logs/judge_%j.out
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_deepmath/ge6_test1000_anchor_bridge_h256_n8x8_seed20260910/logs/judge_%j.err

set -euo pipefail
ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
EXP=${ROOT}/opd_eval_deepmath/ge6_test1000_anchor_bridge_h256_n8x8_seed20260910
MODULE=${ROOT}/ocrd_pipeline/anchor_selector
PYTHON=/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python
test -f "${EXP}/anchor_candidates/GPU_SCORING_COMPLETE"
set -a
source "${MODULE}/.secrets.openlux.env"
set +a
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
curl -fsS --max-time 60 -H "Authorization: Bearer ${OPENLUX_API_KEY}" https://api.openlux.ai/v1/models >/dev/null
"${PYTHON}" "${MODULE}/run_anchor_judge_batch.py" \
  --input-root "${EXP}/anchor_candidates" \
  --output-dir "${EXP}/anchors_luna" \
  --tokenizer /storage/yukaichengLab/share/model/Qwen3-1.7B \
  --config "${MODULE}/config_gpt56_luna_openlux.json" \
  --secrets "${MODULE}/.secrets.openlux.env" \
  --workers 64 --candidate-batch-size 4
test -f "${EXP}/anchors_luna/BATCH_JUDGE_COMPLETE"
