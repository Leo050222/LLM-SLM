#!/bin/bash
#SBATCH --job-name=opd_dm_ge6_fullrkl
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=384G
#SBATCH --gres=gpu:H800:8
#SBATCH --time=7-00:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/opd_dm_ge6_fullrkl.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/opd_dm_ge6_fullrkl.%j.err

set -euo pipefail

source /storage/yukaichengLab/wangxin/miniconda3/etc/profile.d/conda.sh
conda activate verl-opd

export PYTHONPATH=/storage/yukaichengLab/wangxin/test_eiloeo/verl
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export NCCL_DEBUG=WARN
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# Each rollout rank already has exactly one visible GPU.  Running the V1
# EngineCore in-process removes a second process boundary that can terminate
# silently during four concurrent long-context generations; it does not change
# SamplingParams or the per-example seeds.
export VLLM_ENABLE_V1_MULTIPROCESSING=0

SCRIPT=/storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/run_deepmath_full_vocab_reverse_kl_train.py
RUN_DIR=/storage/yukaichengLab/wangxin/test_eiloeo/opd_runs/qwen3_1_7b_from_14b_deepmath_ge6_full_vocab_rkl
CHECKPOINT_ROOT=/storage/yukaichengLab/wangxin/test_eiloeo/opd_checkpoints/qwen3_1_7b_from_14b_deepmath_ge6_full_vocab_rkl
TOTAL_STEPS=885
export FULL_RKL_CHECKPOINT_ROOT=${CHECKPOINT_ROOT}

mkdir -p "${RUN_DIR}" "${CHECKPOINT_ROOT}"
python "${SCRIPT}" init --run-dir "${RUN_DIR}"

student_pid=""
teacher_pids=()
rollout_pids=()
cleanup() {
  status=$?
  for pid in "${teacher_pids[@]:-}" "${rollout_pids[@]:-}"; do
    if [ -n "${pid}" ]; then kill "${pid}" 2>/dev/null || true; fi
  done
  if [ -n "${student_pid}" ]; then kill "${student_pid}" 2>/dev/null || true; fi
  exit "${status}"
}
trap cleanup EXIT INT TERM

CUDA_VISIBLE_DEVICES=0,1,2,3 \
  torchrun --standalone --nproc_per_node=4 "${SCRIPT}" student-daemon \
    --run-dir "${RUN_DIR}" --checkpoint-root "${CHECKPOINT_ROOT}" &
student_pid=$!

for _ in $(seq 1 1800); do
  if [ -f "${RUN_DIR}/student_daemon_ready.json" ]; then break; fi
  if ! kill -0 "${student_pid}" 2>/dev/null; then
    wait "${student_pid}"
    exit 30
  fi
  sleep 2
done
test -f "${RUN_DIR}/student_daemon_ready.json"

start_step=1
if [ -f "${CHECKPOINT_ROOT}/latest_checkpointed_iteration.txt" ]; then
  start_step=$(( $(tr -d '[:space:]' < "${CHECKPOINT_ROOT}/latest_checkpointed_iteration.txt") + 1 ))
fi

for step in $(seq "${start_step}" "${TOTAL_STEPS}"); do
  step_dir=$(printf "%s/steps/step_%04d" "${RUN_DIR}" "${step}")
  if [ -e "${step_dir}" ]; then
    echo "Refusing to overwrite incomplete step directory: ${step_dir}" >&2
    exit 31
  fi
  mkdir -p "${step_dir}"

  rollout_pids=()
  mkdir -p "${step_dir}/rollout_logs"
  for rank in 0 1 2 3; do
    port=$((25000 + (SLURM_JOB_ID % 1000) * 10 + rank))
    CUDA_VISIBLE_DEVICES=${rank} VLLM_PORT=${port} \
      python "${SCRIPT}" rollout --run-dir "${RUN_DIR}" \
        --step "${step}" --rank "${rank}" --model-dir "${RUN_DIR}/current_hf" \
        > "${step_dir}/rollout_logs/rank${rank}.log" \
        2> "${step_dir}/rollout_logs/rank${rank}.err" &
    rollout_pids+=("$!")
  done
  rollout_status=0
  for rank in 0 1 2 3; do
    pid=${rollout_pids[${rank}]}
    if wait "${pid}"; then
      status=0
    else
      status=$?
      rollout_status=1
    fi
    echo "${status}" > "${step_dir}/rollout_logs/rank${rank}.exit_status"
    echo "FULL_VOCAB_RKL_ROLLOUT_EXIT step=${step} rank=${rank} status=${status}"
  done
  rollout_pids=()
  if [ "${rollout_status}" -ne 0 ]; then
    touch "${step_dir}/FAILED"
    exit 32
  fi
  python "${SCRIPT}" merge-rollout --run-dir "${RUN_DIR}" --step "${step}"

  teacher_pids=()
  for rank in 0 1 2 3; do
    gpu=$((rank + 4))
    (
      CUDA_VISIBLE_DEVICES=${gpu} python "${SCRIPT}" teacher \
        --run-dir "${RUN_DIR}" --step "${step}" --rank "${rank}" || {
          mkdir -p "${step_dir}/teacher/rank${rank}"
          touch "${step_dir}/teacher/rank${rank}/FAILED"
          exit 1
        }
    ) &
    teacher_pids+=("$!")
  done
  teacher_status=0
  for pid in "${teacher_pids[@]}"; do wait "${pid}" || teacher_status=1; done
  teacher_pids=()
  if [ "${teacher_status}" -ne 0 ]; then
    touch "${step_dir}/FAILED"
    exit 33
  fi
  if ! kill -0 "${student_pid}" 2>/dev/null; then
    wait "${student_pid}"
    exit 34
  fi

  python "${SCRIPT}" step-summary --run-dir "${RUN_DIR}" --step "${step}"
  if [ "${step}" -lt "${TOTAL_STEPS}" ] && ! kill -0 "${student_pid}" 2>/dev/null; then
    wait "${student_pid}"
    exit 35
  fi
done

wait "${student_pid}"
student_pid=""
python "${SCRIPT}" verify --run-dir "${RUN_DIR}" --checkpoint-root "${CHECKPOINT_ROOT}"
trap - EXIT INT TERM
echo "DEEPMATH_FULL_VOCAB_RKL_FORMAL_JOB_OK job=${SLURM_JOB_ID}"
