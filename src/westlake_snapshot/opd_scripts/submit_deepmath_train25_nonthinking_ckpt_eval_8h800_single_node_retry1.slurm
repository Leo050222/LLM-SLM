#!/bin/bash
#SBATCH --job-name=dm25_nt_ckpteval_8h
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --gres=gpu:H800:8
#SBATCH --exclusive
#SBATCH --exclude=gnho006,gnho009,gnho017,gnho018,gnho032,gnho034,gnho038
#SBATCH --time=24:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/dm25_nt_ckpteval_8h.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/dm25_nt_ckpteval_8h.%j.err

set -euo pipefail
source /storage/yukaichengLab/wangxin/miniconda3/etc/profile.d/conda.sh
conda activate verl-opd

ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
CKPT_ROOT=${ROOT}/opd_checkpoints/qwen3_1_7b_from_14b_deepmath_ge6_train25pct_nonthinking_student_top16_rkl_8h800_single_node
EVAL_ROOT=${ROOT}/opd_eval_deepmath/ge6_test1000_qwen3_1_7b_train25pct_nonthinking_top16_rkl_seed20260811
DATA_ROOT=${ROOT}/opd_data/deepmath_ge6_opd_train15pct_seed20260811/eval_shards4
TOKENIZER=/storage/yukaichengLab/share/model/Qwen3-1.7B
SCRIPT=${ROOT}/opd_scripts/run_opd_logit_diagnostic.py
LOG_ROOT=${ROOT}/opd_logs/dm25_nt_ckpteval_8h.${SLURM_JOB_ID}.workers
STEPS=(44 89 133 178 222)

export PYTHONPATH=${ROOT}/verl
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

test -s "${TOKENIZER}/tokenizer_config.json"
mkdir -p "${EVAL_ROOT}/attempts" "${LOG_ROOT}"
for step in "${STEPS[@]}"; do
  model=${CKPT_ROOT}/global_step_${step}/actor/huggingface
  test -s "${model}/config.json"
  test -s "${model}/model.safetensors.index.json"
done
for shard in $(seq 0 3); do
  test -s "${DATA_ROOT}/test_1000.shard${shard}.parquet"
done

echo "DEEPMATH_TRAIN25_NONTHINKING_CKPTEVAL_START job=${SLURM_JOB_ID} node=${SLURMD_NODENAME} time=$(date --iso-8601=seconds)"
/soft/slurm/bin/scontrol show job "${SLURM_JOB_ID}"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
gpu_count=$(nvidia-smi -L | wc -l)
test "${gpu_count}" -eq 8
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -v 'NVIDIA H800' >/dev/null; then
  echo "Allocated node contains a non-H800 GPU" >&2
  exit 4
fi
other_jobs=$(/soft/slurm/bin/squeue -h -w "${SLURMD_NODENAME}" -o '%A' | sort -u | grep -vx "${SLURM_JOB_ID}" || true)
if [[ -n "${other_jobs}" ]]; then
  echo "Exclusive-node invariant failed: ${other_jobs}" >&2
  exit 5
fi
echo "SINGLE_NODE_EXCLUSIVE_OK node=${SLURMD_NODENAME} gpu_count=${gpu_count}"

pids=()
for rank in $(seq 0 7); do
  (
    set -euo pipefail
    export CUDA_VISIBLE_DEVICES=${rank}
    export OMP_NUM_THREADS=6
    export XDG_CACHE_HOME=/tmp/xdg_dm25_nt_eval_${SLURM_JOB_ID}_${rank}
    export VLLM_CACHE_ROOT=/tmp/vllm_dm25_nt_eval_${SLURM_JOB_ID}_${rank}
    export TRITON_CACHE_DIR=/tmp/triton_dm25_nt_eval_${SLURM_JOB_ID}_${rank}
    export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_dm25_nt_eval_${SLURM_JOB_ID}_${rank}
    mkdir -p "${XDG_CACHE_HOME}" "${VLLM_CACHE_ROOT}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"
    python -c 'import torch; assert torch.cuda.is_available() and torch.cuda.device_count()==1; x=torch.arange(8,device="cuda"); assert int(x.sum())==28; print("CUDA_PREFLIGHT_OK",torch.cuda.get_device_name(),torch.version.cuda)'

    for task in "${rank}" "$((rank + 8))" "$((rank + 16))"; do
      [[ "${task}" -lt 20 ]] || continue
      checkpoint_index=$((task / 4))
      shard=$((task % 4))
      step=${STEPS[${checkpoint_index}]}
      model=${CKPT_ROOT}/global_step_${step}/actor/huggingface
      data=${DATA_ROOT}/test_1000.shard${shard}.parquet
      run=${EVAL_ROOT}/step_${step}
      final=${run}/shards/shard_${shard}
      attempt=${EVAL_ROOT}/attempts/job_${SLURM_JOB_ID}_task_${task}_step_${step}_shard_${shard}
      if [[ -e "${final}" || -e "${attempt}" ]]; then
        echo "Refusing existing output: ${final} ${attempt}" >&2
        exit 6
      fi
      python -c 'import pandas as pd,sys; f=pd.read_parquet(sys.argv[1]); assert len(f)==250; assert all(int(x)%4==int(sys.argv[2]) for x in f["dataset_index"])' "${data}" "${shard}"

      seed_port=$((20000 + (SLURM_JOB_ID % 1000) * 32 + task))
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
      echo "VLLM_PORT_SELECTED rank=${rank} task=${task} step=${step} shard=${shard} selected=${VLLM_PORT}"

      mkdir -p "$(dirname "${attempt}")"
      python "${SCRIPT}" generate --output-dir "${attempt}" \
        --student "${model}" --tokenizer "${TOKENIZER}" \
        --teacher /storage/yukaichengLab/share/model/Qwen3-14B \
        --dataset-name deepmath_ge6_test1000 --data "${data}" \
        --prompt-style deepmath --scorer math --score-timeout 20 \
        --seed 20260811 --max-tokens 40264 --max-model-len 40960 --topk 16 \
        --disable-thinking
      mkdir -p "$(dirname "${final}")"
      mv "${attempt}" "${final}"
      echo "CHECKPOINT_SHARD_OK rank=${rank} task=${task} step=${step} shard=${shard} time=$(date --iso-8601=seconds)"
    done
    echo "WORKER_OK rank=${rank} time=$(date --iso-8601=seconds)"
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
test "${failed}" -eq 0
for step in "${STEPS[@]}"; do
  for shard in $(seq 0 3); do
    test -s "${EVAL_ROOT}/step_${step}/shards/shard_${shard}/rollouts.jsonl"
    test -s "${EVAL_ROOT}/step_${step}/shards/shard_${shard}/run_config.json"
  done
done
echo "DEEPMATH_TRAIN25_NONTHINKING_CKPTEVAL_ALL_SHARDS_OK job=${SLURM_JOB_ID} time=$(date --iso-8601=seconds)"
