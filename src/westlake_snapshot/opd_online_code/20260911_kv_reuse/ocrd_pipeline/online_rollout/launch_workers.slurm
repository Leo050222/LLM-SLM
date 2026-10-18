#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:H800:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=08:00:00
#SBATCH -J online_k8

set -euo pipefail
export PATH=/soft/slurm/bin:$PATH
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=8
export VLLM_WORKER_MULTIPROC_METHOD=spawn
: "${OPD_CODE_ROOT:?Set OPD_CODE_ROOT to the deployed repository root}"
: "${OPD_ONLINE_RUN:?Set OPD_ONLINE_RUN under /storage/yukaichengLab/wangxin/test_eiloeo}"
: "${OPD_PYTHON:?Set OPD_PYTHON to the validated vLLM environment python}"
case "$OPD_ONLINE_RUN" in
  /storage/yukaichengLab/wangxin/test_eiloeo/*) ;;
  *) echo 'Run directory must be under test_eiloeo' >&2; exit 1 ;;
esac
mkdir -p "$OPD_ONLINE_RUN/queue" "$OPD_ONLINE_RUN/logs"
if [[ -f "$OPD_ONLINE_RUN/queue/STOP" ]]; then
  echo 'STOP exists; remove it explicitly before restarting this queue' >&2
  exit 1
fi
export XDG_CACHE_HOME="$OPD_ONLINE_RUN/cache/xdg"
export TORCHINDUCTOR_CACHE_DIR="$OPD_ONLINE_RUN/cache/torchinductor"
export TRITON_CACHE_DIR="$OPD_ONLINE_RUN/cache/triton"
mkdir -p "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
pids=()
cleanup() {
  for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
  wait || true
}
trap cleanup EXIT INT TERM
for role in student teacher; do
  if [[ "$role" == student ]]; then
    model=${OPD_STUDENT_MODEL:-/storage/yukaichengLab/share/model/Qwen3-1.7B}
  else
    model=${OPD_TEACHER_MODEL:-/storage/yukaichengLab/share/model/Qwen3-14B}
  fi
  srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:H800:1 --cpus-per-task=16 --mem=120G \
    "$OPD_PYTHON" "$OPD_CODE_ROOT/ocrd_pipeline/online_rollout/worker.py" \
    --queue "$OPD_ONLINE_RUN/queue" --role "$role" --model "$model" \
    >"$OPD_ONLINE_RUN/logs/${SLURM_JOB_ID}_${role}.log" 2>&1 &
  pids+=("$!")
done
# Exit promptly if either resident worker dies, releasing the other GPU.
while kill -0 "${pids[0]}" 2>/dev/null && kill -0 "${pids[1]}" 2>/dev/null; do
  sleep 2
done
failed=0
for pid in "${pids[@]}"; do
  if ! kill -0 "$pid" 2>/dev/null; then wait "$pid" || failed=1; fi
done
exit "$failed"
