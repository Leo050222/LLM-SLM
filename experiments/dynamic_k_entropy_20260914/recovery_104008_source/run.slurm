#!/bin/bash
#SBATCH -J dynamic_k_entropy
#SBATCH -p yukaichenglab
#SBATCH --account=yukaicheng
#SBATCH --nodes=1
#SBATCH --nodelist=gnho020
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:H800:4
#SBATCH --mem=320G
#SBATCH --time=24:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/dynamic_k_entropy.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/dynamic_k_entropy.%j.err
set -Eeuo pipefail
ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
export OPD_PYTHON=/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python
export SCALE_CODE=$ROOT/opd_scripts/dynamic_k_entropy_20260914
export OPD_ONLINE_RUN=$ROOT/opd_eval_dynamic/dynamic_k_entropy_20260914/slots/job_$SLURM_JOB_ID
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$OPD_ONLINE_RUN/logs"
IFS=',' read -ra devices <<< "${CUDA_VISIBLE_DEVICES:?}"
test "${#devices[@]}" -eq 4
pids=()
cleanup(){ for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done; }
trap cleanup EXIT
for rank in 0 1 2 3; do
 VLLM_PORT=$((20000+rank*512)) SLURM_PROCID=$rank CUDA_VISIBLE_DEVICES=${devices[$rank]} bash "$SCALE_CODE/worker_rank.sh" > "$OPD_ONLINE_RUN/logs/gpu${rank}.log" 2>&1 &
 pids+=("$!")
done
"$OPD_PYTHON" -u "$SCALE_CODE/dispatch.py" > "$OPD_ONLINE_RUN/logs/dispatch.log" 2>&1 &
coordinator=$!
while kill -0 "$coordinator" 2>/dev/null; do
 for pid in "${pids[@]}"; do
  if ! kill -0 "$pid" 2>/dev/null; then
   test -f "$ROOT/opd_eval_dynamic/dynamic_k_entropy_20260914/ALL_GENERATION_COMPLETE.json" || { kill "$coordinator"; exit 1; }
  fi
 done
 sleep 5
done
wait "$coordinator"
for pid in "${pids[@]}"; do wait "$pid"; done
