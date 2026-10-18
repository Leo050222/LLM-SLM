#!/bin/bash
#SBATCH --job-name=dm_train25_14b_nt_smoke
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=128G
#SBATCH --gres=gpu:H800:1
#SBATCH --exclude=gnho009,gnho018
#SBATCH --time=00:45:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/dm_train25_14b_nt_smoke.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/dm_train25_14b_nt_smoke.%j.err

set -euo pipefail
source /storage/yukaichengLab/wangxin/miniconda3/etc/profile.d/conda.sh
conda activate verl-opd

ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
RUN=${ROOT}/opd_rollouts/deepmath_ge6_train25pct_qwen3_14b_native_nonthinking_seed20260811
INPUT=${RUN}/inputs/shard_00.parquet
OUTPUT=${RUN}/smoke/job_${SLURM_JOB_ID}
MODEL=/storage/yukaichengLab/share/model/Qwen3-14B
TOKENIZER=/storage/yukaichengLab/share/model/Qwen3-1.7B
SCRIPT=${ROOT}/opd_scripts/run_deepmath_train25_nonthinking_rollout.py

export PYTHONPATH=${ROOT}/verl
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=8 VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton_dm_train25_14b_nt_smoke_${SLURM_JOB_ID}
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_dm_train25_14b_nt_smoke_${SLURM_JOB_ID}
mkdir -p "${RUN}/smoke" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"
if [[ -e "${OUTPUT}" ]]; then
  echo "Refusing existing smoke output: ${OUTPUT}" >&2
  exit 3
fi

echo "DEEPMATH_TRAIN25_QWEN3_14B_NATIVE_NONTHINKING_SMOKE_START job=${SLURM_JOB_ID} node=${SLURMD_NODENAME} time=$(date --iso-8601=seconds)"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
python -c 'import torch; assert torch.cuda.is_available(); x=torch.arange(8,device="cuda"); assert int(x.sum())==28; print("CUDA_PREFLIGHT_OK",torch.cuda.get_device_name(),torch.version.cuda)'

seed_port=$((22000 + (SLURM_JOB_ID % 1000) * 16))
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

SHARD_SHA=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["shards"][0]["sha256"])' "${RUN}/prepare_manifest.json")
python "${SCRIPT}" \
  --input "${INPUT}" --input-sha256 "${SHARD_SHA}" \
  --output-dir "${OUTPUT}" \
  --model "${MODEL}" --tokenizer "${TOKENIZER}" \
  --shard-index 0 --num-shards 16 --expected-rows 888 --limit 1 \
  --seed 20260811 --max-tokens 40264 --max-model-len 40960
python -c 'import json,sys; c=json.load(open(sys.argv[1])); assert c["status"]=="DEEPMATH_TRAIN25_NATIVE_NONTHINKING_SHARD_OK"; assert c["model"]==sys.argv[2]; assert c["tokenizer"]==sys.argv[3]; assert c["num_prompts"]==1; assert c["thinking_mode"] is False; print("DEEPMATH_TRAIN25_QWEN3_14B_NATIVE_NONTHINKING_SMOKE_OK")' "${OUTPUT}/run_config.json" "${MODEL}" "${TOKENIZER}"
