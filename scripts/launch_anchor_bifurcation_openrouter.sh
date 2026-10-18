#!/bin/bash
set -euo pipefail
umask 077

ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
SCRIPT=${ROOT}/opd_scripts/run_anchor_bifurcation_openrouter.py
INPUT=${ROOT}/opd_analysis/anchor_bifurcation_h256_openrouter_luna_seed20260908/inputs/judge_inputs.jsonl
OUT=${ROOT}/opd_analysis/anchor_bifurcation_h256_openrouter_luna_seed20260908/openrouter_run
LOG=${ROOT}/opd_logs/anchor_bifurcation_openrouter_luna_20260908.log
SESSION=anchor_bifurcation_luna

test -s "${SCRIPT}"
test -s "${INPUT}"
if screen -ls | grep -F ".${SESSION}" >/dev/null; then
  echo "Refusing duplicate screen session: ${SESSION}" >&2
  exit 3
fi
if ps -eo args | grep -E '[r]un_anchor_bifurcation_openrouter.py' >/dev/null; then
  echo "Refusing duplicate OpenRouter runner" >&2
  exit 4
fi
if [[ -e "${OUT}/ANCHOR_BIFURCATION_OPENROUTER_RUN_COMPLETE" ]]; then
  echo "Refusing to overwrite completed run: ${OUT}" >&2
  exit 5
fi

IFS= read -r -s OPENROUTER_API_KEY
echo
test -n "${OPENROUTER_API_KEY}"
export OPENROUTER_API_KEY
mkdir -p "${OUT}"

echo "OPENROUTER_PILOT_START time=$(date --iso-8601=seconds) model=openai/gpt-5.6-luna items=24 workers=24"
python "${SCRIPT}" \
  --inputs "${INPUT}" --output-dir "${OUT}" \
  --model openai/gpt-5.6-luna --workers 24 --limit 24 \
  --timeout 300 --attempts 6 2>&1 | tee "${LOG}.pilot"
echo "OPENROUTER_PILOT_OK time=$(date --iso-8601=seconds)"

screen -dmS "${SESSION}" bash -c "exec python '${SCRIPT}' --inputs '${INPUT}' --output-dir '${OUT}' --model openai/gpt-5.6-luna --workers 96 --timeout 300 --attempts 8 >'${LOG}' 2>&1"
unset OPENROUTER_API_KEY
sleep 2
screen -ls | grep -F ".${SESSION}"
echo "OPENROUTER_FULL_RUN_LAUNCHED session=${SESSION} workers=96 time=$(date --iso-8601=seconds) log=${LOG}"
