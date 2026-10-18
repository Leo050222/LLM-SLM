#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=1024G
#SBATCH --gres=gpu:H800:8
#SBATCH --exclusive
#SBATCH --exclude=gnho009,gnho018
#SBATCH --time=24:00:00
#SBATCH -J q35judge_full
#SBATCH -o /storage/yukaichengLab/wangxin/test_eiloeo/qwen35_27b_anchor_judge_full/logs/%j.out
#SBATCH -e /storage/yukaichengLab/wangxin/test_eiloeo/qwen35_27b_anchor_judge_full/logs/%j.err

set -euo pipefail

ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
MODULE=${ROOT}/ocrd_pipeline/anchor_selector
INPUT=${ROOT}/ocrd_anchor_train25_qwen3_1_7b_nonthinking_seed20260811_transformers_v2
OUTPUT=${INPUT}/judge_local_qwen35_27b_full
RUN_ROOT=${ROOT}/qwen35_27b_anchor_judge_full
MODEL=/storage/yukaichengLab/share/model/Qwen3.5-27B
TOKENIZER=/storage/yukaichengLab/share/model/Qwen3-1.7B
PYTHON=/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python
HTTP_OVERLAY=${ROOT}/env_overlays/qwen35_judge_httpfix
BASE_PORT=19000

mkdir -p ${RUN_ROOT}/logs ${OUTPUT}
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONPATH=${HTTP_OVERLAY}:${PYTHONPATH:-}

${PYTHON} ${MODULE}/verify_qwen35_judge_deployment.py

declare -a SERVER_PIDS=()
cleanup() {
  if [[ -n "${MONITOR_PID:-}" ]]; then kill ${MONITOR_PID} 2>/dev/null || true; fi
  for pid in "${SERVER_PIDS[@]:-}"; do
    if [[ -n "${pid}" ]]; then kill ${pid} 2>/dev/null || true; fi
  done
}
trap cleanup EXIT

(
  while true; do
    date +%s
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader,nounits
    sleep 10
  done
) > ${RUN_ROOT}/logs/${SLURM_JOB_ID}.gpu.csv 2>&1 &
MONITOR_PID=$!

for gpu in $(seq 0 7); do
  port=$((BASE_PORT + gpu))
  (
    export CUDA_VISIBLE_DEVICES=${gpu}
    export VLLM_CACHE_ROOT=/tmp/q35judge_${SLURM_JOB_ID}/gpu_${gpu}/vllm
    export XDG_CACHE_HOME=/tmp/q35judge_${SLURM_JOB_ID}/gpu_${gpu}/xdg
    export TRITON_CACHE_DIR=/tmp/q35judge_${SLURM_JOB_ID}/gpu_${gpu}/triton
    mkdir -p ${VLLM_CACHE_ROOT} ${XDG_CACHE_HOME} ${TRITON_CACHE_DIR}
    exec ${PYTHON} -m vllm.entrypoints.openai.api_server \
      --model ${MODEL} \
      --served-model-name qwen3.5-27b-anchor-judge \
      --host 127.0.0.1 \
      --port ${port} \
      --dtype bfloat16 \
      --max-model-len 49152 \
      --gpu-memory-utilization 0.92 \
      --max-num-seqs 32 \
      --enable-prefix-caching \
      --gdn-prefill-backend triton \
      --no-enable-log-requests
  ) > ${RUN_ROOT}/logs/${SLURM_JOB_ID}.server_gpu${gpu}.out \
    2> ${RUN_ROOT}/logs/${SLURM_JOB_ID}.server_gpu${gpu}.err &
  SERVER_PIDS[${gpu}]=$!
done

for gpu in $(seq 0 7); do
  port=$((BASE_PORT + gpu))
  READY=0
  for _ in $(seq 1 240); do
    if ${PYTHON} -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${port}/health',timeout=2).read()" >/dev/null 2>&1; then
      READY=1
      break
    fi
    if ! kill -0 ${SERVER_PIDS[${gpu}]} 2>/dev/null; then
      echo "SERVER_EXITED_BEFORE_READY gpu=${gpu}"
      exit 1
    fi
    sleep 5
  done
  if [[ ${READY} -ne 1 ]]; then
    echo "SERVER_READY_TIMEOUT gpu=${gpu}"
    exit 1
  fi
  echo "SERVER_READY gpu=${gpu} port=${port} pid=${SERVER_PIDS[${gpu}]}"
done

echo "ALL_SERVERS_READY job=${SLURM_JOB_ID} node=${SLURMD_NODENAME}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader

START=$(date +%s)
${PYTHON} ${MODULE}/run_anchor_judge_batch.py \
  --input-root ${INPUT} \
  --output-dir ${OUTPUT} \
  --tokenizer ${TOKENIZER} \
  --config ${MODULE}/config_qwen35_27b_local_8gpu.json \
  --workers 48 \
  --candidate-batch-size 4
END=$(date +%s)

echo "FULL_JUDGE_COMPLETE elapsed_seconds=$((END-START))"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
