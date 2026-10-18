#!/usr/bin/env bash
set -euo pipefail

: "${OPD_SCRIPT:?}"
: "${OPD_RUN_DIR:?}"
: "${OPD_CHECKPOINT_ROOT:?}"
: "${OPD_TOTAL_STEPS:?}"

if [[ "${SLURM_NNODES}" -ne 1 || "${OPD_STUDENT_WORLD_SIZE:-4}" -ne 4 ]]; then
  echo "Expected one node and four student ranks" >&2
  exit 20
fi

mkdir -p "${OPD_RUN_DIR}" "${OPD_CHECKPOINT_ROOT}" "${OPD_RUN_DIR}/daemon_logs"
python "${OPD_SCRIPT}" init --run-dir "${OPD_RUN_DIR}" \
  --checkpoint-root "${OPD_CHECKPOINT_ROOT}"

mapfile -t hosts < <(/soft/slurm/bin/scontrol show hostnames "${SLURM_JOB_NODELIST}")
if [[ "${#hosts[@]}" -ne 1 ]]; then
  echo "Expected exactly one allocated host" >&2
  exit 21
fi
export OPD_MASTER_ADDR=${hosts[0]}
export OPD_MASTER_PORT=$((30000 + SLURM_JOB_ID % 1000))

student_pid=""
teacher_pid=""
cleanup() {
  rc=$?
  for pid in "${student_pid}" "${teacher_pid}"; do
    if [[ -n "${pid}" ]]; then
      kill "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  done
  exit "${rc}"
}
trap cleanup EXIT INT TERM

/soft/slurm/bin/srun --overlap --kill-on-bad-exit=1 --nodes=1 --ntasks=1 \
  --cpus-per-task=32 bash -lc '
    set -euo pipefail
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    torchrun --nnodes=1 --nproc_per_node=4 --node_rank=0 \
      --master_addr="${OPD_MASTER_ADDR}" --master_port="${OPD_MASTER_PORT}" \
      "${OPD_SCRIPT}" student-daemon --run-dir "${OPD_RUN_DIR}" \
      --checkpoint-root "${OPD_CHECKPOINT_ROOT}"
  ' >"${OPD_RUN_DIR}/daemon_logs/student.out" \
    2>"${OPD_RUN_DIR}/daemon_logs/student.err" &
student_pid=$!

/soft/slurm/bin/srun --overlap --kill-on-bad-exit=1 --nodes=1 --ntasks=1 \
  --cpus-per-task=24 bash -lc '
    set -euo pipefail
    pids=()
    failed=0
    mkdir -p "${OPD_RUN_DIR}/daemon_logs/teacher_status"
    for rank in 0 1 2 3; do
      gpu=$((rank + 4))
      CUDA_VISIBLE_DEVICES=${gpu} python "${OPD_SCRIPT}" teacher-daemon \
        --run-dir "${OPD_RUN_DIR}" --checkpoint-root "${OPD_CHECKPOINT_ROOT}" \
        --rank "${rank}" \
        >"${OPD_RUN_DIR}/daemon_logs/teacher_rank${rank}.out" \
        2>"${OPD_RUN_DIR}/daemon_logs/teacher_rank${rank}.err" &
      pids+=("$!")
    done
    for rank in 0 1 2 3; do
      set +e
      wait "${pids[$rank]}"
      rc=$?
      set -e
      printf "rank\t%s\npid\t%s\nexit_code\t%s\n" \
        "${rank}" "${pids[$rank]}" "${rc}" \
        >"${OPD_RUN_DIR}/daemon_logs/teacher_status/rank${rank}.exit"
      if [[ "${rc}" -ne 0 ]]; then failed=1; fi
    done
    exit "${failed}"
  ' >"${OPD_RUN_DIR}/daemon_logs/teacher_parent.out" \
    2>"${OPD_RUN_DIR}/daemon_logs/teacher_parent.err" &
teacher_pid=$!

daemon_alive() {
  local pid=$1 name=$2 state
  state=$(ps -o stat= -p "${pid}" 2>/dev/null | tr -d '[:space:]' || true)
  if [[ -z "${state}" || "${state}" == Z* ]]; then
    set +e
    wait "${pid}"
    rc=$?
    set -e
    echo "${name} daemon exited before expected marker, exit=${rc}" >&2
    return 1
  fi
}

for _ in $(seq 1 1800); do
  student_ready=0
  teacher_ready=0
  [[ -f "${OPD_RUN_DIR}/student_daemon_ready.json" ]] && student_ready=1
  teacher_ready=$(find "${OPD_RUN_DIR}/teacher_ready" -type f -name 'rank*.ready' 2>/dev/null | wc -l || true)
  if [[ "${student_ready}" -eq 1 && "${teacher_ready}" -eq 4 ]]; then break; fi
  daemon_alive "${student_pid}" student
  daemon_alive "${teacher_pid}" teacher
  sleep 2
done
test -f "${OPD_RUN_DIR}/student_daemon_ready.json"
test "$(find "${OPD_RUN_DIR}/teacher_ready" -type f -name 'rank*.ready' | wc -l)" -eq 4

start_step=1
if [[ -f "${OPD_CHECKPOINT_ROOT}/latest_checkpointed_iteration.txt" ]]; then
  start_step=$(( $(tr -d '[:space:]' <"${OPD_CHECKPOINT_ROOT}/latest_checkpointed_iteration.txt") + 1 ))
fi

for step in $(seq "${start_step}" "${OPD_TOTAL_STEPS}"); do
  export OPD_STEP=${step}
  step_dir=$(printf "%s/steps/step_%04d" "${OPD_RUN_DIR}" "${step}")
  python "${OPD_SCRIPT}" prepare-step --run-dir "${OPD_RUN_DIR}" \
    --checkpoint-root "${OPD_CHECKPOINT_ROOT}" --step "${step}"
  while [[ ! -f "${step_dir}/STUDENT_DONE" ]]; do
    daemon_alive "${student_pid}" student || exit 30
    daemon_alive "${teacher_pid}" teacher || exit 31
    sleep 2
  done
  for rank in 0 1 2 3; do
    test -f "${step_dir}/TEACHER_RANK_${rank}_DONE"
  done
  python "${OPD_SCRIPT}" step-summary --run-dir "${OPD_RUN_DIR}" \
    --checkpoint-root "${OPD_CHECKPOINT_ROOT}" --step "${step}"
done

set +e
wait "${student_pid}"
student_rc=$?
student_pid=""
wait "${teacher_pid}"
teacher_rc=$?
teacher_pid=""
set -e
printf 'student_exit_code\t%s\nteacher_exit_code\t%s\n' \
  "${student_rc}" "${teacher_rc}" >"${OPD_RUN_DIR}/daemon_logs/final_exit_status.tsv"
if [[ "${student_rc}" -ne 0 || "${teacher_rc}" -ne 0 ]]; then
  exit 32
fi

python "${OPD_SCRIPT}" verify --run-dir "${OPD_RUN_DIR}" \
  --checkpoint-root "${OPD_CHECKPOINT_ROOT}"
trap - EXIT INT TERM
echo "BRIDGE_TOP16_8H800_SINGLE_NODE_JOB_OK job=${SLURM_JOB_ID}"
