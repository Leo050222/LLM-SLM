#!/bin/bash
set -euo pipefail

ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
RUN=${ROOT}/ocrd_anchor_train25_qwen3_1_7b_nonthinking_seed20260811_transformers_v2
MODULE=${ROOT}/ocrd_pipeline/anchor_selector
OUTPUT=${RUN}/judge_remote_gpt56_luna_batch4
PYTHON=/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python
LOG=${OUTPUT}/logs/login_judge_resume_20260824.log
PID_FILE=${OUTPUT}/logs/login_judge_resume_20260824.pid

mkdir -p "${OUTPUT}/logs"
/soft/slurm/bin/scancel 92943 2>/dev/null || true
if [[ -s "${PID_FILE}" ]] && kill -0 "$(<"${PID_FILE}")" 2>/dev/null; then
  echo "LUNA_LOGIN_ALREADY_RUNNING pid=$(<"${PID_FILE}")"
  exit 0
fi
nohup setsid bash -lc "
  export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
  ${PYTHON} ${MODULE}/run_anchor_judge_batch.py \\
    --input-root ${RUN} \\
    --output-dir ${OUTPUT} \\
    --tokenizer /storage/yukaichengLab/share/model/Qwen3-1.7B \\
    --config ${MODULE}/config_gpt56_luna.json \\
    --secrets ${MODULE}/.secrets.env \\
    --workers 64 \\
    --candidate-batch-size 4 \\
    > ${LOG} 2>&1
" >/dev/null 2>&1 < /dev/null &
pid=$!
printf '%s\n' "${pid}" > "${PID_FILE}"

sleep 3
if kill -0 "${pid}" 2>/dev/null; then
  echo "LUNA_LOGIN_RUNNING pid=${pid}"
else
  echo "LUNA_LOGIN_FAILED pid=${pid}"
  exit 1
fi
tail -n 20 "${LOG}" 2>/dev/null || true
