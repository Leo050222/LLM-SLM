#!/bin/bash
set -euo pipefail

source /storage/yukaichengLab/wangxin/miniconda3/etc/profile.d/conda.sh
conda activate verl-opd

ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
MODEL_PATH=/storage/yukaichengLab/share/model/Qwen3-1.7B
SERVED_MODEL=Qwen3-1.7B
DATA=${ROOT}/opd_eval_math500/data/AI-ModelScope_MATH-500/test.jsonl
DATA_SHA256=35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132
WORK_DIR=${ROOT}/opd_eval_math500/results/qwen3_1_7b_baseline_seed20260811
SERVER_LOG=${WORK_DIR}/vllm_server.log

export PYTHONPATH=${ROOT}/eval_repos/evalscope
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 MODELSCOPE_OFFLINE=1
export OMP_NUM_THREADS=8 VLLM_WORKER_MULTIPROC_METHOD=spawn
export TRITON_CACHE_DIR=/tmp/triton_math500_qwen17b_${SLURM_JOB_ID}
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_math500_qwen17b_${SLURM_JOB_ID}

test "$(sha256sum "${DATA}" | awk '{print $1}')" = "${DATA_SHA256}"
test "$(wc -l < "${DATA}")" -eq 500
if [ -e "${WORK_DIR}/reports" ] || [ -e "${SERVER_LOG}" ]; then
  echo "Refusing to overwrite existing MATH-500 baseline output: ${WORK_DIR}" >&2
  exit 3
fi
mkdir -p "${WORK_DIR}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"

seed_port=$((18000 + (SLURM_JOB_ID % 5000)))
PORT=""
for offset in $(seq 0 99); do
  candidate=$((seed_port + offset))
  if python -c 'import socket,sys; s=socket.socket(); s.bind(("127.0.0.1", int(sys.argv[1]))); s.close()' "${candidate}" 2>/dev/null; then
    PORT=${candidate}
    break
  fi
done
test -n "${PORT}"

echo "MATH500_BASELINE_START job=${SLURM_JOB_ID} node=${SLURMD_NODENAME} port=${PORT}"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
python "${ROOT}/opd_scripts/run_vllm_api_no_metrics.py" \
  --model "${MODEL_PATH}" --served-model-name "${SERVED_MODEL}" \
  --host 127.0.0.1 --port "${PORT}" --dtype bfloat16 \
  --tensor-parallel-size 1 --max-model-len 40960 --max-num-seqs 32 \
  --gpu-memory-utilization 0.92 --enable-prefix-caching \
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
    tail -200 "${SERVER_LOG}" >&2
    exit 4
  fi
  if curl --fail --silent "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
    break
  fi
  sleep 2
done
curl --fail --silent "http://127.0.0.1:${PORT}/v1/models" >/dev/null

python "${ROOT}/opd_scripts/run_qwen3_math500_baseline.py" \
  --model-id "${SERVED_MODEL}" \
  --api-url "http://127.0.0.1:${PORT}/v1/chat/completions" \
  --work-dir "${WORK_DIR}"

echo "QWEN3_1_7B_MATH500_BASELINE_OK job=${SLURM_JOB_ID} end=$(date --iso-8601=seconds)"
