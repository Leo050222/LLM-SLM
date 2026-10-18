#!/bin/bash
#SBATCH --job-name=dm_train25_nt_rollout
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=128G
#SBATCH --gres=gpu:H800:1
#SBATCH --exclude=gnho009,gnho018
#SBATCH --array=0-15%16
#SBATCH --time=06:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/dm_train25_nt_rollout.%A_%a.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/dm_train25_nt_rollout.%A_%a.err

set -euo pipefail
source /storage/yukaichengLab/wangxin/miniconda3/etc/profile.d/conda.sh
conda activate verl-opd

ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
RUN=${ROOT}/opd_rollouts/deepmath_ge6_train25pct_qwen3_1_7b_native_nonthinking_seed20260811
TASK=${SLURM_ARRAY_TASK_ID}
INPUT=${RUN}/inputs/shard_$(printf '%02d' "${TASK}").parquet
FINAL=${RUN}/shards/shard_$(printf '%02d' "${TASK}")
ATTEMPT=${RUN}/attempts/job_${SLURM_ARRAY_JOB_ID}_task_$(printf '%02d' "${TASK}")
MODEL=/storage/yukaichengLab/share/model/Qwen3-1.7B
TOKENIZER=/storage/yukaichengLab/share/model/Qwen3-1.7B
SCRIPT=${ROOT}/opd_scripts/run_deepmath_train25_nonthinking_rollout.py

export PYTHONPATH=${ROOT}/verl
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=8 VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton_dm_train25_nt_${SLURM_ARRAY_JOB_ID}_${TASK}
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_dm_train25_nt_${SLURM_ARRAY_JOB_ID}_${TASK}

test -s "${MODEL}/config.json"
test -s "${TOKENIZER}/tokenizer_config.json"
test -s "${INPUT}"
test -f "${RUN}/prepare_manifest.json"
if [[ -e "${FINAL}" || -e "${ATTEMPT}" ]]; then
  echo "Refusing existing final/attempt output: ${FINAL} ${ATTEMPT}" >&2
  exit 3
fi
mkdir -p "${RUN}/attempts" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"

PROBE=${RUN}/.write_probe_${SLURM_ARRAY_JOB_ID}_${TASK}
trap 'unlink "${PROBE}" 2>/dev/null || true' EXIT
dd if=/dev/zero of="${PROBE}" bs=1M count=1 status=none
unlink "${PROBE}"

echo "DEEPMATH_TRAIN25_NATIVE_NONTHINKING_START array=${SLURM_ARRAY_JOB_ID} task=${TASK} node=${SLURMD_NODENAME} time=$(date --iso-8601=seconds)"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
python -c 'import torch; assert torch.cuda.is_available(); x=torch.arange(8,device="cuda"); assert int(x.sum())==28; print("CUDA_PREFLIGHT_OK",torch.cuda.get_device_name(),torch.version.cuda)'

seed_port=$((20000 + (SLURM_ARRAY_JOB_ID % 1000) * 16 + TASK))
VLLM_PORT=""
for offset in $(seq 0 127); do
  candidate=$((seed_port + offset * 16))
  if python -c 'import socket,sys; s=socket.socket(); s.bind(("127.0.0.1",int(sys.argv[1]))); s.close()' "${candidate}" 2>/dev/null; then
    VLLM_PORT=${candidate}
    break
  fi
done
test -n "${VLLM_PORT}"
export VLLM_PORT
echo "VLLM_PORT_SELECTED seed=${seed_port} selected=${VLLM_PORT}"

SHARD_SHA=$(python -c 'import json,sys; m=json.load(open(sys.argv[1])); print(m["shards"][int(sys.argv[2])]["sha256"])' "${RUN}/prepare_manifest.json" "${TASK}")
python "${SCRIPT}" \
  --input "${INPUT}" --input-sha256 "${SHARD_SHA}" \
  --output-dir "${ATTEMPT}" \
  --model "${MODEL}" --tokenizer "${TOKENIZER}" \
  --shard-index "${TASK}" --num-shards 16 --expected-rows 888 \
  --seed 20260811 --max-tokens 40264 --max-model-len 40960
mv "${ATTEMPT}" "${FINAL}"
echo "DEEPMATH_TRAIN25_NATIVE_NONTHINKING_SHARD_OK array=${SLURM_ARRAY_JOB_ID} task=${TASK} time=$(date --iso-8601=seconds)"
