#!/bin/bash
set -euo pipefail

MODEL_TAG=${1:?model tag required: qwen3_4b or qwen3_14b}
LIMIT=${2:-}

case "${MODEL_TAG}" in
  qwen3_4b)
    MODEL_PATH=/storage/yukaichengLab/share/model/Qwen3-4B
    SERVED_MODEL=Qwen3-4B
    ;;
  qwen3_14b)
    MODEL_PATH=/storage/yukaichengLab/share/model/Qwen3-14B
    SERVED_MODEL=Qwen3-14B
    ;;
  *)
    echo "Unknown model tag: ${MODEL_TAG}" >&2
    exit 2
    ;;
esac

source /storage/yukaichengLab/wangxin/miniconda3/etc/profile.d/conda.sh
conda activate verl-opd

ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
EVALSCOPE_ROOT=${ROOT}/eval_repos/evalscope
RUNNER=${ROOT}/opd_scripts/run_qwen3_hard_eval.py
OUTPUT_ROOT=${ROOT}/opd_eval_hard/results
PORT=$((8800 + ${SLURM_ARRAY_TASK_ID:-0}))
RUN_KIND=full
if [[ -n "${LIMIT}" ]]; then
  RUN_KIND=smoke_limit_${LIMIT}
fi
WORK_DIR=${OUTPUT_ROOT}/${RUN_KIND}/${MODEL_TAG}
SERVER_LOG=${WORK_DIR}/vllm_server.log

export PYTHONPATH=${EVALSCOPE_ROOT}
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export MODELSCOPE_OFFLINE=1
export OMP_NUM_THREADS=8
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TRITON_CACHE_DIR=/tmp/triton_qwen_hard_${SLURM_JOB_ID}_${MODEL_TAG}
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_qwen_hard_${SLURM_JOB_ID}_${MODEL_TAG}

mkdir -p "${WORK_DIR}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Array task: ${SLURM_ARRAY_TASK_ID:-none}"
echo "Node: ${SLURMD_NODENAME}"
echo "Model: ${SERVED_MODEL} (${MODEL_PATH})"
echo "Run kind: ${RUN_KIND}"
echo "Start time: $(date --iso-8601=seconds)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

python "${ROOT}/opd_scripts/run_vllm_api_no_metrics.py" \
  --model "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL}" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-model-len 40960 \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.92 \
  --enable-prefix-caching \
  >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

cleanup() {
  if kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 180); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "vLLM server exited during startup" >&2
    tail -n 200 "${SERVER_LOG}" >&2
    exit 1
  fi
  if curl --fail --silent "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
    break
  fi
  sleep 2
done
curl --fail --silent "http://127.0.0.1:${PORT}/v1/models" >/dev/null
echo "SERVER_READY port=${PORT} time=$(date --iso-8601=seconds)"

LIMIT_ARGS=()
if [[ -n "${LIMIT}" ]]; then
  LIMIT_ARGS=(--limit "${LIMIT}")
fi
python "${RUNNER}" \
  --model-id "${SERVED_MODEL}" \
  --api-url "http://127.0.0.1:${PORT}/v1/chat/completions" \
  --work-dir "${WORK_DIR}" \
  "${LIMIT_ARGS[@]}"

echo "EVAL_COMPLETE model=${SERVED_MODEL} time=$(date --iso-8601=seconds)"
