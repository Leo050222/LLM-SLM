#!/bin/bash
#SBATCH --job-name=dm_ge6_native_nt
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --gres=gpu:H800:1
#SBATCH --exclude=gnho018
#SBATCH --array=0-7%8
#SBATCH --time=12:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/dm_ge6_native_nt.%A_%a.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/dm_ge6_native_nt.%A_%a.err

set -euo pipefail
source /storage/yukaichengLab/wangxin/miniconda3/etc/profile.d/conda.sh
conda activate verl-opd

ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
TASK=${SLURM_ARRAY_TASK_ID}
MODEL_INDEX=$((TASK / 4))
SHARD=$((TASK % 4))
MODEL_LABELS=(qwen3_1_7b qwen3_14b)
MODELS=(/storage/yukaichengLab/share/model/Qwen3-1.7B /storage/yukaichengLab/share/model/Qwen3-14B)
MODEL_LABEL=${MODEL_LABELS[${MODEL_INDEX}]}
MODEL=${MODELS[${MODEL_INDEX}]}
TOKENIZER=/storage/yukaichengLab/share/model/Qwen3-1.7B
DATA=${ROOT}/opd_data/deepmath_ge6_opd_train15pct_seed20260811/eval_shards4/test_1000.shard${SHARD}.parquet
RUN=${ROOT}/opd_eval_deepmath/ge6_test1000_native_nonthinking_seed20260811/${MODEL_LABEL}
OUT=${RUN}/shards/shard_${SHARD}
SCRIPT=${ROOT}/opd_scripts/run_opd_logit_diagnostic.py

export PYTHONPATH=${ROOT}/verl
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=8 VLLM_WORKER_MULTIPROC_METHOD=spawn PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton_dm_ge6_native_nt_${SLURM_ARRAY_JOB_ID}_${TASK}
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_dm_ge6_native_nt_${SLURM_ARRAY_JOB_ID}_${TASK}

if [[ ! -s "${MODEL}/config.json" || ! -s "${TOKENIZER}/tokenizer_config.json" ]]; then
  echo "Incomplete model or tokenizer: ${MODEL} ${TOKENIZER}" >&2
  exit 2
fi
if [[ -e "${OUT}" ]]; then
  echo "Refusing to overwrite shard output: ${OUT}" >&2
  exit 3
fi
python -c 'import pandas as pd,sys; f=pd.read_parquet(sys.argv[1]); assert len(f)==250; assert all(int(x)%4==int(sys.argv[2]) for x in f["dataset_index"])' "${DATA}" "${SHARD}"

echo "DEEPMATH_GE6_NATIVE_NONTHINKING_START array=${SLURM_ARRAY_JOB_ID} task=${TASK} model=${MODEL_LABEL} shard=${SHARD} node=${SLURMD_NODENAME} time=$(date --iso-8601=seconds)"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
python -c 'import torch; assert torch.cuda.is_available(); x=torch.arange(8,device="cuda"); print("CUDA_PREFLIGHT_OK",torch.cuda.get_device_name(),int(x.sum()))'

seed_port=$((20000 + (SLURM_ARRAY_JOB_ID % 1000) * 8 + TASK))
VLLM_PORT=""
for offset in $(seq 0 99); do
  candidate=$((seed_port + offset * 8))
  if python -c 'import socket,sys; s=socket.socket(); s.bind(("127.0.0.1",int(sys.argv[1]))); s.close()' "${candidate}" 2>/dev/null; then
    VLLM_PORT=${candidate}
    break
  fi
done
test -n "${VLLM_PORT}"
export VLLM_PORT
echo "VLLM_PORT_SELECTED seed=${seed_port} selected=${VLLM_PORT}"

mkdir -p "${OUT}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"
python "${SCRIPT}" generate --output-dir "${OUT}" \
  --student "${MODEL}" --tokenizer "${TOKENIZER}" \
  --teacher /storage/yukaichengLab/share/model/Qwen3-14B \
  --dataset-name deepmath_ge6_test1000 --data "${DATA}" \
  --prompt-style deepmath --scorer math --score-timeout 20 \
  --seed 20260811 --max-tokens 40264 --max-model-len 40960 --topk 16 \
  --disable-thinking
echo "DEEPMATH_GE6_NATIVE_NONTHINKING_SHARD_OK model=${MODEL_LABEL} shard=${SHARD} time=$(date --iso-8601=seconds)"
