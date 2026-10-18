#!/bin/bash
#SBATCH --job-name=teacher8_student8
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=640G
#SBATCH --gres=gpu:H800:8
#SBATCH --time=24:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/t8s8.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/t8s8.%j.err
set -Eeuo pipefail
trap 'rc=$?; echo "BATCH_ERR rc=$rc line=$LINENO command=$BASH_COMMAND" >&2' ERR
trap 'echo "BATCH_SIGNAL_TERM" >&2; exit 143' TERM
source /storage/yukaichengLab/wangxin/miniconda3/etc/profile.d/conda.sh
conda activate verl-opd
ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
SCRIPT=${ROOT}/opd_scripts/teacher8_student8_20260909/pipeline.py
OUT=${1:-${ROOT}/opd_eval_deepmath/teacher8_student8_20260909}
LIMIT=${2:-1000}
export PYTHONPATH=${ROOT}/opd_scripts:${ROOT}/verl
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
IFS=',' read -r -a devices <<< "${CUDA_VISIBLE_DEVICES}"
N=${#devices[@]}
test "$N" -le 8 && test "$N" -ge 1
/soft/slurm/bin/scontrol show job "$SLURM_JOB_ID"
python "$SCRIPT" prepare --output "$OUT" --limit "$LIMIT"
for stage in teacher student; do
 pids=()
 for ((rank=0;rank<N;rank++)); do
  (
   trap 'rc=$?; echo "WORKER_EXIT stage=$stage rank=$rank rc=$rc time=$(date -Is)"' EXIT
   export CUDA_VISIBLE_DEVICES=${devices[$rank]}
   export VLLM_CACHE_ROOT=/tmp/t8s8_${SLURM_JOB_ID}_${rank}_${stage}
   export TRITON_CACHE_DIR=/tmp/triton_t8s8_${SLURM_JOB_ID}_${rank}_${stage}
   export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_t8s8_${SLURM_JOB_ID}_${rank}_${stage}
   export VLLM_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')
   setsid --wait python "$SCRIPT" "$stage" --output "$OUT" --limit "$LIMIT" --rank "$rank" --shards "$N"
  ) >"${ROOT}/opd_logs/t8s8.${SLURM_JOB_ID}.${stage}.${rank}.log" 2>&1 &
  pids+=("$!")
 done
 failed=0
 for ((rank=0;rank<N;rank++)); do
  if wait "${pids[$rank]}"; then echo "WORKER_OK $stage $rank"; else echo "WORKER_FAILED $stage $rank"; failed=1; fi
 done
 test "$failed" -eq 0
 python "$SCRIPT" "verify-${stage}" --output "$OUT" --limit "$LIMIT"
 sleep 10
done
echo GENERATION_ALL_COMPLETE
