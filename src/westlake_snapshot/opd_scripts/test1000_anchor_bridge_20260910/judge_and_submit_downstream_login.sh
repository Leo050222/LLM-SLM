#!/bin/bash
set -euo pipefail

ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
EXP=${ROOT}/opd_eval_deepmath/ge6_test1000_anchor_bridge_h256_n8x8_seed20260910
MODULE=${ROOT}/ocrd_pipeline/anchor_selector
DIR=${ROOT}/opd_scripts/test1000_anchor_bridge_20260910
PYTHON=/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python
OUTPUT=${EXP}/anchors_luna
LOG=${EXP}/logs/judge_login_retry_$(date +%Y%m%d_%H%M%S).log

test -f "${EXP}/anchor_candidates/GPU_SCORING_COMPLETE"
if [[ ! -f "${OUTPUT}/BATCH_JUDGE_COMPLETE" ]]; then
  export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
  "${PYTHON}" "${MODULE}/run_anchor_judge_batch.py" \
    --input-root "${EXP}/anchor_candidates" \
    --output-dir "${OUTPUT}" \
    --tokenizer /storage/yukaichengLab/share/model/Qwen3-1.7B \
    --config "${MODULE}/config_gpt56_luna_openlux.json" \
    --secrets "${MODULE}/.secrets.openlux.env" \
    --workers 8 --candidate-batch-size 1 \
    >"${LOG}" 2>&1
fi
test -f "${OUTPUT}/BATCH_JUDGE_COMPLETE"

if [[ -f "${EXP}/downstream_job_ids.env" ]]; then
  echo "Refusing duplicate downstream submission: ${EXP}/downstream_job_ids.env" >&2
  exit 7
fi
bridge=$(/soft/slurm/bin/sbatch --parsable "${DIR}/bridges_node8.slurm")
rollout=$(/soft/slurm/bin/sbatch --parsable --dependency=afterok:${bridge} "${DIR}/rollout_replicate_node8.slurm")
score=$(/soft/slurm/bin/sbatch --parsable --dependency=afterok:${rollout} "${DIR}/score_replicate_cpu.slurm")
aggregate=$(/soft/slurm/bin/sbatch --parsable --dependency=afterok:${score} "${DIR}/aggregate.slurm")
temporary=${EXP}/downstream_job_ids.env.tmp.$$
printf 'BRIDGE=%s\nROLLOUT_ARRAY=%s\nSCORE_ARRAY=%s\nAGGREGATE=%s\n' \
  "${bridge}" "${rollout}" "${score}" "${aggregate}" >"${temporary}"
mv "${temporary}" "${EXP}/downstream_job_ids.env"
cat "${EXP}/downstream_job_ids.env"
