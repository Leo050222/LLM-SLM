#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:H800:1
#SBATCH --exclude=gnho009,gnho018
#SBATCH --time=01:00:00
#SBATCH -J q35judge_smoke
#SBATCH -o /storage/yukaichengLab/wangxin/test_eiloeo/qwen35_27b_anchor_judge_smoke/logs/%j.out
#SBATCH -e /storage/yukaichengLab/wangxin/test_eiloeo/qwen35_27b_anchor_judge_smoke/logs/%j.err

set -euo pipefail

ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
MODULE=${ROOT}/ocrd_pipeline/anchor_selector
INPUT=${ROOT}/ocrd_anchor_train25_qwen3_1_7b_nonthinking_seed20260811_transformers_v2
OUTPUT=${INPUT}/judge_local_qwen35_27b_smoke20_v2
SMOKE=${ROOT}/qwen35_27b_anchor_judge_smoke
MODEL=/storage/yukaichengLab/share/model/Qwen3.5-27B
TOKENIZER=/storage/yukaichengLab/share/model/Qwen3-1.7B
PYTHON=/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python
HTTP_OVERLAY=${ROOT}/env_overlays/qwen35_judge_httpfix
PORT=19000

mkdir -p ${SMOKE}/logs ${OUTPUT}
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONPATH=${HTTP_OVERLAY}:${PYTHONPATH:-}
export VLLM_CACHE_ROOT=/tmp/q35judge_${SLURM_JOB_ID}/vllm
export XDG_CACHE_HOME=/tmp/q35judge_${SLURM_JOB_ID}/xdg
export TRITON_CACHE_DIR=/tmp/q35judge_${SLURM_JOB_ID}/triton
mkdir -p ${VLLM_CACHE_ROOT} ${XDG_CACHE_HOME} ${TRITON_CACHE_DIR}

cleanup() {
  if [[ -n "${MONITOR_PID:-}" ]]; then kill ${MONITOR_PID} 2>/dev/null || true; fi
  if [[ -n "${SERVER_PID:-}" ]]; then kill ${SERVER_PID} 2>/dev/null || true; fi
}
trap cleanup EXIT

echo "SMOKE_START job=${SLURM_JOB_ID} node=${SLURMD_NODENAME}"
${PYTHON} -c 'import torch; torch.manual_seed(0); x=torch.randn(8,8,device="cuda"); print("WITNESS", (x@x).shape, torch.cuda.get_device_name())'
${PYTHON} -c 'import torch,transformers,vllm; print("VERSIONS", torch.__version__, transformers.__version__, vllm.__version__)'

(
  while true; do
    date +%s
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader,nounits
    sleep 5
  done
) > ${SMOKE}/logs/${SLURM_JOB_ID}.gpu.csv 2>&1 &
MONITOR_PID=$!

CUDA_VISIBLE_DEVICES=0 ${PYTHON} -m vllm.entrypoints.openai.api_server \
  --model ${MODEL} \
  --served-model-name qwen3.5-27b-anchor-judge \
  --host 127.0.0.1 \
  --port ${PORT} \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 32 \
  --enable-prefix-caching \
  --gdn-prefill-backend triton \
  --no-enable-log-requests \
  > ${SMOKE}/logs/${SLURM_JOB_ID}.server.out \
  2> ${SMOKE}/logs/${SLURM_JOB_ID}.server.err &
SERVER_PID=$!

READY=0
for _ in $(seq 1 180); do
  if ${PYTHON} -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${PORT}/health',timeout=2).read()" >/dev/null 2>&1; then
    READY=1
    break
  fi
  if ! kill -0 ${SERVER_PID} 2>/dev/null; then
    echo "SERVER_EXITED_BEFORE_READY"
    exit 1
  fi
  sleep 5
done
if [[ ${READY} -ne 1 ]]; then
  echo "SERVER_READY_TIMEOUT"
  exit 1
fi

echo "SERVER_READY pid=${SERVER_PID}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader

START=$(date +%s)
${PYTHON} ${MODULE}/run_anchor_judge_batch.py \
  --input-root ${INPUT} \
  --output-dir ${OUTPUT} \
  --tokenizer ${TOKENIZER} \
  --config ${MODULE}/config_qwen35_27b_local_smoke.json \
  --workers 8 \
  --candidate-batch-size 4 \
  --max-records 20
END=$(date +%s)

echo "SMOKE_COMPLETE elapsed_seconds=$((END-START))"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
