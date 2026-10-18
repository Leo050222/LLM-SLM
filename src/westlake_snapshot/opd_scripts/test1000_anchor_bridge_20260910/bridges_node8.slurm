#!/bin/bash
#SBATCH --job-name=t1k_bridge256
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --gres=gpu:H800:8
#SBATCH --exclusive
#SBATCH --exclude=gnho009,gnho018,gnho036
#SBATCH --time=03:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_deepmath/ge6_test1000_anchor_bridge_h256_n8x8_seed20260910/logs/bridge_%j.out
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_deepmath/ge6_test1000_anchor_bridge_h256_n8x8_seed20260910/logs/bridge_%j.err

set -euo pipefail
ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
EXP=${ROOT}/opd_eval_deepmath/ge6_test1000_anchor_bridge_h256_n8x8_seed20260910
MODULE=${ROOT}/ocrd_pipeline/bridge_generator
PYTHON=/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python
ANCHORS=${EXP}/anchors_luna
OUT=${EXP}/bridges_h256_n8
MODEL=/storage/yukaichengLab/share/model/Qwen3-14B
test -f "${ANCHORS}/BATCH_JUDGE_COMPLETE"
mkdir -p "${OUT}/logs"
pids=()
for shard_id in $(seq 0 7); do
  (
    set -euo pipefail
    export CUDA_VISIBLE_DEVICES=${shard_id}
    export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
    export OMP_NUM_THREADS=6 VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_WORKER_MULTIPROC_METHOD=spawn
    export VLLM_PORT=$((24000 + (SLURM_JOB_ID % 1000) * 8 + shard_id))
    export XDG_CACHE_HOME=/tmp/t1k_bridge_${SLURM_JOB_ID}_${shard_id}/xdg
    export VLLM_CACHE_ROOT=/tmp/t1k_bridge_${SLURM_JOB_ID}_${shard_id}/vllm
    export TORCHINDUCTOR_CACHE_DIR=/tmp/t1k_bridge_${SLURM_JOB_ID}_${shard_id}/torchinductor
    export TRITON_CACHE_DIR=/tmp/t1k_bridge_${SLURM_JOB_ID}_${shard_id}/triton
    mkdir -p "${XDG_CACHE_HOME}" "${VLLM_CACHE_ROOT}" "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"
    "${PYTHON}" "${MODULE}/run_bridge_batch.py" --anchor-root "${ANCHORS}" --output-root "${OUT}" \
      --shard-id "${shard_id}" --num-shards 8 --h 256 --teacher "${MODEL}" \
      --temperature 1.0 --top-p 1.0 --top-k 0 --seed 20260822 --first-step-logprobs 16 \
      --max-model-len 4096 --num-rollouts 8 --trace-batch-size 8 --gpu-memory-utilization 0.90
  ) >"${OUT}/logs/${SLURM_JOB_ID}_shard_${shard_id}.out" 2>"${OUT}/logs/${SLURM_JOB_ID}_shard_${shard_id}.err" &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
test "${failed}" -eq 0
selected=$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_anchors"])' "${ANCHORS}/summary.json")
touch "${OUT}/GPU_GENERATION_COMPLETE"
"${PYTHON}" "${MODULE}/verify_bridge_batch.py" --anchor-root "${ANCHORS}" --output-root "${OUT}" \
  --num-shards 8 --h 256 --num-rollouts 8 --teacher "${MODEL}" --expected-count "${selected}"
