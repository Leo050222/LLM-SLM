#!/bin/bash
#SBATCH --job-name=dm_ge6_fullrkl_smoke
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=384G
#SBATCH --gres=gpu:H800:8
#SBATCH --time=12:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/dm_ge6_fullrkl_smoke.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/dm_ge6_fullrkl_smoke.%j.err

set -euo pipefail

source /storage/yukaichengLab/wangxin/miniconda3/etc/profile.d/conda.sh
conda activate verl-opd

export PYTHONPATH=/storage/yukaichengLab/wangxin/test_eiloeo/verl
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_DEVICE_ORDER=PCI_BUS_ID

SCRIPT=/storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/run_full_vocab_reverse_kl_smoke.py
OUTPUT_ROOT=/storage/yukaichengLab/wangxin/test_eiloeo/opd_smoke/deepmath_ge6_full_vocab_rkl_maxlen
OUTPUT_DIR=${OUTPUT_ROOT}/job_${SLURM_JOB_ID}
CHUNK_TOKENS=256

if [ -e "${OUTPUT_DIR}" ]; then
  echo "Refusing to overwrite existing smoke output: ${OUTPUT_DIR}" >&2
  exit 20
fi
mkdir -p "${OUTPUT_DIR}"

available_kb=$(df -Pk "${OUTPUT_ROOT}" | awk 'NR==2 {print $4}')
if [ "${available_kb}" -lt 104857600 ]; then
  echo "At least 100 GiB free space is required for full-logit smoke" >&2
  exit 21
fi

python "${SCRIPT}" unit --output-dir "${OUTPUT_DIR}" --chunk-tokens "${CHUNK_TOKENS}"

generate_pids=()
for rank in 0 1 2 3; do
  port=$((24000 + (SLURM_JOB_ID % 1000) * 10 + rank))
  CUDA_VISIBLE_DEVICES=${rank} VLLM_PORT=${port} \
    python "${SCRIPT}" generate \
      --rank "${rank}" --output-dir "${OUTPUT_DIR}" \
      --chunk-tokens "${CHUNK_TOKENS}" &
  generate_pids+=("$!")
done
generate_status=0
for pid in "${generate_pids[@]}"; do
  wait "${pid}" || generate_status=1
done
if [ "${generate_status}" -ne 0 ]; then
  echo "At least one rollout process failed" >&2
  exit 22
fi

pair_pids=()
for rank in 0 1 2 3; do
  teacher_gpu=$((rank + 4))
  (
    CUDA_VISIBLE_DEVICES=${teacher_gpu} \
      python "${SCRIPT}" teacher \
        --rank "${rank}" --output-dir "${OUTPUT_DIR}" \
        --chunk-tokens "${CHUNK_TOKENS}" || {
          mkdir -p "${OUTPUT_DIR}/teacher_logits/rank${rank}"
          touch "${OUTPUT_DIR}/teacher_logits/rank${rank}/FAILED"
          exit 1
        }
  ) &
  pair_pids+=("$!")

  CUDA_VISIBLE_DEVICES=${rank} \
    python "${SCRIPT}" student \
      --rank "${rank}" --output-dir "${OUTPUT_DIR}" \
      --chunk-tokens "${CHUNK_TOKENS}" \
      --teacher-wait-seconds 21600 &
  pair_pids+=("$!")
done
pair_status=0
for pid in "${pair_pids[@]}"; do
  wait "${pid}" || pair_status=1
done
if [ "${pair_status}" -ne 0 ]; then
  echo "At least one teacher/student full-vocabulary process failed" >&2
  exit 23
fi

python "${SCRIPT}" summarize \
  --output-dir "${OUTPUT_DIR}" --chunk-tokens "${CHUNK_TOKENS}"

test -f "${OUTPUT_DIR}/summary.json"
grep -q 'DEEPMATH_FULL_VOCAB_RKL_MAXLEN_SMOKE_OK' "${OUTPUT_DIR}/summary.json"
echo "DEEPMATH_FULL_VOCAB_RKL_MAXLEN_SMOKE_JOB_OK job=${SLURM_JOB_ID} output=${OUTPUT_DIR}"
