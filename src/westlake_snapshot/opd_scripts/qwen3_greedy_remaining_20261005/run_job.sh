#!/bin/bash
set -Eeuo pipefail

MODEL_TAG=${1:?model tag required: qwen3_4b or qwen3_8b}
case "${MODEL_TAG}" in
  qwen3_4b) MODEL_PATH=/storage/yukaichengLab/share/model/Qwen3-4B; SERVED_MODEL=Qwen3-4B ;;
  qwen3_8b) MODEL_PATH=/storage/yukaichengLab/share/model/Qwen3-8B; SERVED_MODEL=Qwen3-8B ;;
  *) echo "unknown model tag: ${MODEL_TAG}" >&2; exit 2 ;;
esac

source /storage/yukaichengLab/wangxin/miniconda3/etc/profile.d/conda.sh
conda activate verl-opd
ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
RUNNER=${ROOT}/opd_scripts/qwen3_greedy_remaining_20261005
INPUT_ROOT=${ROOT}/opd_eval_baselines/qwen3_greedy_remaining_20261005
RUN_ROOT=${INPUT_ROOT}/${MODEL_TAG}
PORT=$((8900 + ${SLURM_JOB_ID:-0} % 100))
SERVER_LOG=${RUN_ROOT}/vllm_server.log
mkdir -p "${RUN_ROOT}" "/tmp/triton_qwen3_remaining_${SLURM_JOB_ID}" "/tmp/torchinductor_qwen3_remaining_${SLURM_JOB_ID}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 MODELSCOPE_OFFLINE=1
export OMP_NUM_THREADS=8 VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONPATH=${ROOT}/eval_repos/evalscope:${PYTHONPATH:-}
export TRITON_CACHE_DIR=/tmp/triton_qwen3_remaining_${SLURM_JOB_ID}
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_qwen3_remaining_${SLURM_JOB_ID}

echo "JOB=${SLURM_JOB_ID} NODE=${SLURMD_NODENAME} MODEL=${SERVED_MODEL} START=$(date --iso-8601=seconds)"
python "${RUNNER}/prepare.py"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
python "${ROOT}/opd_scripts/run_vllm_api_no_metrics.py" \
  --model "${MODEL_PATH}" --served-model-name "${SERVED_MODEL}" \
  --host 127.0.0.1 --port "${PORT}" --dtype bfloat16 \
  --tensor-parallel-size 8 --max-model-len 40960 --max-num-seqs 32 \
  --gpu-memory-utilization 0.92 >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
cleanup() { kill "${SERVER_PID}" 2>/dev/null || true; wait "${SERVER_PID}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
for _ in $(seq 1 180); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    tail -n 200 "${SERVER_LOG}" >&2
    exit 1
  fi
  if curl --fail --silent "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then break; fi
  sleep 2
done
curl --fail --silent "http://127.0.0.1:${PORT}/v1/models" >/dev/null
echo "SERVER_READY port=${PORT} time=$(date --iso-8601=seconds)"
python "${RUNNER}/run_greedy.py" --model-path "${MODEL_PATH}" --model-name "${SERVED_MODEL}" \
  --input "${INPUT_ROOT}/prompts.jsonl" --output "${RUN_ROOT}" --port "${PORT}"
python "${RUNNER}/score_extra.py" --output "${RUN_ROOT}"
echo "EVAL_COMPLETE model=${SERVED_MODEL} time=$(date --iso-8601=seconds)"
