#!/bin/bash
set -euo pipefail

: "${OPD_SCRIPT:?}"
: "${OPD_RUN_DIR:?}"
: "${OPD_CHECKPOINT_ROOT:?}"
: "${OPD_TOTAL_STEPS:?}"
: "${OPD_STUDENT_WORLD_SIZE:?}"

if [ "${SLURM_NNODES}" -ne 2 ] || [ "${OPD_STUDENT_WORLD_SIZE}" -ne 8 ]; then
  echo "Expected two nodes and eight student ranks" >&2
  exit 20
fi

export FULL_RKL_CHECKPOINT_ROOT="${OPD_CHECKPOINT_ROOT}"
mkdir -p "${OPD_RUN_DIR}" "${OPD_CHECKPOINT_ROOT}"
python "${OPD_SCRIPT}" init --run-dir "${OPD_RUN_DIR}"

mapfile -t OPD_HOSTS < <(/soft/slurm/bin/scontrol show hostnames "${SLURM_JOB_NODELIST}")
if [ "${#OPD_HOSTS[@]}" -ne 2 ]; then
  echo "Expected exactly two allocated hosts, got ${#OPD_HOSTS[@]}" >&2
  exit 21
fi
export OPD_MASTER_ADDR="${OPD_HOSTS[0]}"
export OPD_MASTER_PORT=$((29000 + SLURM_JOB_ID % 1000))

student_pid=""
cleanup() {
  status=$?
  if [ -n "${student_pid}" ]; then
    kill "${student_pid}" 2>/dev/null || true
    wait "${student_pid}" 2>/dev/null || true
  fi
  exit "${status}"
}
trap cleanup EXIT INT TERM

/soft/slurm/bin/srun --overlap --kill-on-bad-exit=1 --nodes=2 --ntasks=2 \
  --ntasks-per-node=1 --cpus-per-task=32 bash -lc '
    set -euo pipefail
    node_rank=${SLURM_PROCID}
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    torchrun --nnodes=2 --nproc_per_node=4 --node_rank=${node_rank} \
      --master_addr="${OPD_MASTER_ADDR}" --master_port="${OPD_MASTER_PORT}" \
      "${OPD_SCRIPT}" student-daemon --run-dir "${OPD_RUN_DIR}" \
      --checkpoint-root "${OPD_CHECKPOINT_ROOT}"
  ' &
student_pid=$!

for _ in $(seq 1 1800); do
  if [ -f "${OPD_RUN_DIR}/student_daemon_ready.json" ]; then break; fi
  if ! kill -0 "${student_pid}" 2>/dev/null; then
    wait "${student_pid}"
    exit 30
  fi
  sleep 2
done
test -f "${OPD_RUN_DIR}/student_daemon_ready.json"

start_step=1
if [ -f "${OPD_CHECKPOINT_ROOT}/latest_checkpointed_iteration.txt" ]; then
  start_step=$(( $(tr -d '[:space:]' < "${OPD_CHECKPOINT_ROOT}/latest_checkpointed_iteration.txt") + 1 ))
fi

for step in $(seq "${start_step}" "${OPD_TOTAL_STEPS}"); do
  export OPD_STEP="${step}"
  step_dir=$(printf "%s/steps/step_%04d" "${OPD_RUN_DIR}" "${step}")
  if [ -e "${step_dir}" ]; then
    echo "Refusing to overwrite incomplete step directory: ${step_dir}" >&2
    exit 31
  fi
  mkdir -p "${step_dir}/rollout_logs"

  /soft/slurm/bin/srun --overlap --kill-on-bad-exit=1 --nodes=2 --ntasks=2 \
    --ntasks-per-node=1 --cpus-per-task=24 bash -lc '
      set -euo pipefail
      base=$((SLURM_PROCID * 4))
      pids=()
      statuses=0
      choose_vllm_port() {
        seed_port=$1
        for attempt in $(seq 0 99); do
          candidate=$((seed_port + attempt * 4))
          if python -c "import socket,sys; s=socket.socket(); s.bind((\"127.0.0.1\", int(sys.argv[1]))); s.close()" "${candidate}" 2>/dev/null; then
            printf "%s\n" "${candidate}"
            return 0
          fi
        done
        return 1
      }
      for local_rank in 0 1 2 3; do
        rank=$((base + local_rank))
        # A new rendezvous port per optimizer step avoids TCPStore reuse races
        # when the previous in-process vLLM engine has only just exited. Probe
        # the rank-specific residue class as well, so an unrelated listener on
        # the deterministic first choice cannot abort the optimizer step.
        slot=$(( (SLURM_JOB_ID * 137 + OPD_STEP * 17) % 5000 ))
        seed_port=$((12000 + slot * 8 + local_rank))
        if ! port=$(choose_vllm_port "${seed_port}"); then
          echo "No free vLLM rendezvous port for rank=${rank} seed=${seed_port}" >&2
          exit 41
        fi
        rollout_log="${OPD_RUN_DIR}/steps/$(printf step_%04d "${OPD_STEP}")/rollout_logs/rank${rank}.log"
        printf "VLLM_PORT_SELECTED rank=%s seed=%s selected=%s\n" "${rank}" "${seed_port}" "${port}" > "${rollout_log}"
        CUDA_VISIBLE_DEVICES=${local_rank} VLLM_PORT=${port} \
          python "${OPD_SCRIPT}" rollout --run-dir "${OPD_RUN_DIR}" \
            --step "${OPD_STEP}" --rank "${rank}" --model-dir "${OPD_RUN_DIR}/current_hf" \
            >> "${rollout_log}" \
            2> "${OPD_RUN_DIR}/steps/$(printf step_%04d "${OPD_STEP}")/rollout_logs/rank${rank}.err" &
        pids+=("$!")
      done
      for local_rank in 0 1 2 3; do
        rank=$((base + local_rank))
        if wait "${pids[${local_rank}]}"; then status=0; else status=$?; statuses=1; fi
        echo "${status}" > "${OPD_RUN_DIR}/steps/$(printf step_%04d "${OPD_STEP}")/rollout_logs/rank${rank}.exit_status"
      done
      exit "${statuses}"
    ' || { touch "${step_dir}/FAILED"; exit 32; }

  python "${OPD_SCRIPT}" merge-rollout --run-dir "${OPD_RUN_DIR}" --step "${step}"

  /soft/slurm/bin/srun --overlap --kill-on-bad-exit=1 --nodes=2 --ntasks=2 \
    --ntasks-per-node=1 --cpus-per-task=24 bash -lc '
      set -euo pipefail
      base=$((SLURM_PROCID * 4))
      pids=()
      statuses=0
      for local_rank in 0 1 2 3; do
        rank=$((base + local_rank))
        gpu=$((local_rank + 4))
        CUDA_VISIBLE_DEVICES=${gpu} python "${OPD_SCRIPT}" teacher \
          --run-dir "${OPD_RUN_DIR}" --step "${OPD_STEP}" --rank "${rank}" &
        pids+=("$!")
      done
      for local_rank in 0 1 2 3; do
        if ! wait "${pids[${local_rank}]}"; then statuses=1; fi
      done
      exit "${statuses}"
    ' || { touch "${step_dir}/FAILED"; exit 33; }

  if ! kill -0 "${student_pid}" 2>/dev/null; then
    wait "${student_pid}"
    exit 34
  fi
  python "${OPD_SCRIPT}" step-summary --run-dir "${OPD_RUN_DIR}" --step "${step}"
done

wait "${student_pid}"
student_pid=""
python "${OPD_SCRIPT}" verify --run-dir "${OPD_RUN_DIR}" \
  --checkpoint-root "${OPD_CHECKPOINT_ROOT}"
trap - EXIT INT TERM
echo "DEEPMATH_TOP16_16H800_JOB_OK job=${SLURM_JOB_ID}"
