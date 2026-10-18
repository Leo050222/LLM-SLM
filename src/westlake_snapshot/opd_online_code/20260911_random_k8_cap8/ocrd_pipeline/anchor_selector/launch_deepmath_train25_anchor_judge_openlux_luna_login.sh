#!/bin/bash
set -euo pipefail

ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
RUN=${ROOT}/ocrd_anchor_train25_qwen3_1_7b_nonthinking_seed20260811_transformers_v2
MODULE=${ROOT}/ocrd_pipeline/anchor_selector
OUTPUT=${RUN}/judge_remote_gpt56_luna_batch4
PYTHON=/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python
LOG=${OUTPUT}/logs/login_judge_openlux_resume_attempt3_20260824.log
PID_FILE=${OUTPUT}/logs/login_judge_openlux_resume_attempt3_20260824.pid
EXIT_FILE=${OUTPUT}/logs/login_judge_openlux_resume_attempt3_20260824.exit
ARCHIVE=${OUTPUT}/logs/pre_openlux_resume_20260824_1645

mkdir -p "${OUTPUT}/logs"
if [[ -e "${OUTPUT}/BATCH_JUDGE_COMPLETE" ]]; then
  echo "LUNA_BATCH_ALREADY_COMPLETE"
  exit 0
fi
if [[ -s "${PID_FILE}" ]] && kill -0 "$(<"${PID_FILE}")" 2>/dev/null; then
  echo "OPENLUX_LUNA_ALREADY_RUNNING pid=$(<"${PID_FILE}")"
  exit 0
fi

mkdir -p "${ARCHIVE}"
for name in BILLING_STOPPED summary.json anchors.jsonl judgments.jsonl failures.jsonl; do
  if [[ -e "${OUTPUT}/${name}" ]]; then
    mv "${OUTPUT}/${name}" "${ARCHIVE}/${name}"
  fi
done

nohup setsid bash -lc "
  export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
  set +e
  ${PYTHON} ${MODULE}/run_anchor_judge_batch.py \\
    --input-root ${RUN} \\
    --output-dir ${OUTPUT} \\
    --tokenizer /storage/yukaichengLab/share/model/Qwen3-1.7B \\
    --config ${MODULE}/config_gpt56_luna_openlux.json \\
    --secrets ${MODULE}/.secrets.openlux.env \\
    --workers 64 \\
    --candidate-batch-size 4 \\
    > ${LOG} 2>&1
  status=\$?
  printf '%s\\n' \"\${status}\" > ${EXIT_FILE}
  exit \"\${status}\"
" >/dev/null 2>&1 < /dev/null &
pid=$!
printf '%s\n' "${pid}" > "${PID_FILE}"

sleep 3
if kill -0 "${pid}" 2>/dev/null; then
  echo "OPENLUX_LUNA_RUNNING pid=${pid}"
else
  echo "OPENLUX_LUNA_FAILED pid=${pid}"
  exit 1
fi
tail -n 20 "${LOG}" 2>/dev/null || true
