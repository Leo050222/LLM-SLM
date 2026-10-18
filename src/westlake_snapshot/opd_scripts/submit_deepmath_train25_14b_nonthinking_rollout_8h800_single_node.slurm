#!/bin/bash
#SBATCH --job-name=dm14b_nt_8h_single
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --gres=gpu:H800:8
#SBATCH --exclusive
#SBATCH --exclude=gnho009,gnho018
#SBATCH --time=04:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/dm14b_nt_8h_single.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/dm14b_nt_8h_single.%j.err

set -euo pipefail
source /storage/yukaichengLab/wangxin/miniconda3/etc/profile.d/conda.sh
conda activate verl-opd

ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
RUN=${ROOT}/opd_rollouts/deepmath_ge6_train25pct_qwen3_14b_native_nonthinking_seed20260811_single_node_8h800
MODEL=/storage/yukaichengLab/share/model/Qwen3-14B
TOKENIZER=/storage/yukaichengLab/share/model/Qwen3-1.7B
SCRIPT=${ROOT}/opd_scripts/run_deepmath_train25_nonthinking_rollout.py
LOG_ROOT=${ROOT}/opd_logs/dm14b_nt_8h_single.${SLURM_JOB_ID}.workers

export PYTHONPATH=${ROOT}/verl
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

test -s "${MODEL}/config.json"
test -s "${TOKENIZER}/tokenizer_config.json"
test -f "${RUN}/prepare_manifest.json"
mkdir -p "${RUN}/attempts" "${LOG_ROOT}"

echo "DEEPMATH_TRAIN25_QWEN3_14B_8H800_START job=${SLURM_JOB_ID} node=${SLURMD_NODENAME} time=$(date --iso-8601=seconds)"
/soft/slurm/bin/scontrol show job "${SLURM_JOB_ID}"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader

gpu_count=$(nvidia-smi -L | wc -l)
if [[ "${gpu_count}" -ne 8 ]]; then
  echo "Expected exactly 8 visible GPUs on the exclusive node, got ${gpu_count}" >&2
  exit 4
fi
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -v 'NVIDIA H800' >/dev/null; then
  echo "Allocated node contains a non-H800 GPU" >&2
  exit 5
fi
other_jobs=$(/soft/slurm/bin/squeue -h -w "${SLURMD_NODENAME}" -o '%A' | sort -u | grep -vx "${SLURM_JOB_ID}" || true)
if [[ -n "${other_jobs}" ]]; then
  echo "Exclusive-node invariant failed; other allocations on ${SLURMD_NODENAME}: ${other_jobs}" >&2
  exit 6
fi
echo "SINGLE_NODE_EXCLUSIVE_OK node=${SLURMD_NODENAME} job=${SLURM_JOB_ID} gpu_count=${gpu_count}"

PROBE=${RUN}/.write_probe_${SLURM_JOB_ID}
trap 'unlink "${PROBE}" 2>/dev/null || true' EXIT
dd if=/dev/zero of="${PROBE}" bs=1M count=16 status=none
unlink "${PROBE}"

pids=()
for rank in $(seq 0 7); do
  (
    set -euo pipefail
    export OPD_WORKER_RANK=${rank}
    export CUDA_VISIBLE_DEVICES=${rank}
    export OMP_NUM_THREADS=6
    export XDG_CACHE_HOME=/tmp/xdg_dm14b_nt_${SLURM_JOB_ID}_${rank}
    export VLLM_CACHE_ROOT=/tmp/vllm_dm14b_nt_${SLURM_JOB_ID}_${rank}
    export TRITON_CACHE_DIR=/tmp/triton_dm14b_nt_${SLURM_JOB_ID}_${rank}
    export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_dm14b_nt_${SLURM_JOB_ID}_${rank}
    mkdir -p "${XDG_CACHE_HOME}" "${VLLM_CACHE_ROOT}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"

    python -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count()==1; x=torch.arange(8,device="cuda"); assert int(x.sum())==28; print("CUDA_PREFLIGHT_OK",torch.cuda.get_device_name(),torch.version.cuda)'

    for task in "${rank}" "$((rank + 8))"; do
      input=${RUN}/inputs/shard_$(printf '%02d' "${task}").parquet
      final=${RUN}/shards/shard_$(printf '%02d' "${task}")
      attempt=${RUN}/attempts/job_${SLURM_JOB_ID}_rank_$(printf '%02d' "${rank}")_task_$(printf '%02d' "${task}")
      test -s "${input}"
      if [[ -e "${final}" || -e "${attempt}" ]]; then
        echo "Refusing existing final/attempt output: ${final} ${attempt}" >&2
        exit 7
      fi

      seed_port=$((22000 + (SLURM_JOB_ID % 1000) * 32 + task))
      selected_port=""
      for offset in $(seq 0 127); do
        candidate=$((seed_port + offset * 32))
        if python -c 'import socket,sys; s=socket.socket(); s.bind(("127.0.0.1",int(sys.argv[1]))); s.close()' "${candidate}" 2>/dev/null; then
          selected_port=${candidate}
          break
        fi
      done
      test -n "${selected_port}"
      export VLLM_PORT=${selected_port}
      echo "VLLM_PORT_SELECTED rank=${rank} task=${task} seed=${seed_port} selected=${VLLM_PORT}"

      shard_sha=$(python -c 'import json,sys; m=json.load(open(sys.argv[1])); print(m["shards"][int(sys.argv[2])]["sha256"])' "${RUN}/prepare_manifest.json" "${task}")
      python "${SCRIPT}" \
        --input "${input}" --input-sha256 "${shard_sha}" \
        --output-dir "${attempt}" \
        --model "${MODEL}" --tokenizer "${TOKENIZER}" \
        --shard-index "${task}" --num-shards 16 --expected-rows 888 \
        --seed 20260811 --max-tokens 40264 --max-model-len 40960
      mv "${attempt}" "${final}"
      echo "SHARD_OK rank=${rank} task=${task} node=${SLURMD_NODENAME} time=$(date --iso-8601=seconds)"
    done
    echo "WORKER_OK rank=${rank} node=${SLURMD_NODENAME} time=$(date --iso-8601=seconds)"
  ) >"${LOG_ROOT}/rank_${rank}.log" 2>"${LOG_ROOT}/rank_${rank}.err" &
  pids+=("$!")
done

failed=0
for rank in $(seq 0 7); do
  if ! wait "${pids[$rank]}"; then
    echo "WORKER_FAILED rank=${rank} log=${LOG_ROOT}/rank_${rank}.log err=${LOG_ROOT}/rank_${rank}.err" >&2
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  exit 8
fi

for task in $(seq 0 15); do
  test -f "${RUN}/shards/shard_$(printf '%02d' "${task}")/SHARD_COMPLETE"
done
echo "DEEPMATH_TRAIN25_QWEN3_14B_8H800_ALL_SHARDS_OK job=${SLURM_JOB_ID} node=${SLURMD_NODENAME} time=$(date --iso-8601=seconds)"
