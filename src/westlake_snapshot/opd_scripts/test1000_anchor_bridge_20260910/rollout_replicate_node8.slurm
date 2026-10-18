#!/bin/bash
#SBATCH --job-name=t1k_sbridge8x8
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --gres=gpu:H800:8
#SBATCH --exclusive
#SBATCH --exclude=gnho009,gnho018,gnho036
#SBATCH --array=0-7%3
#SBATCH --time=05:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_deepmath/ge6_test1000_anchor_bridge_h256_n8x8_seed20260910/logs/rollout_%A_%a.out
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_deepmath/ge6_test1000_anchor_bridge_h256_n8x8_seed20260910/logs/rollout_%A_%a.err

set -euo pipefail
ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
EXP=${ROOT}/opd_eval_deepmath/ge6_test1000_anchor_bridge_h256_n8x8_seed20260910
MODULE=${ROOT}/ocrd_pipeline/full_rollout
PYTHON=/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python
MODEL=/storage/yukaichengLab/share/model/Qwen3-1.7B
BRIDGES=${EXP}/bridges_h256_n8
REP=${SLURM_ARRAY_TASK_ID}
OUT=${EXP}/student_after_bridge_n8x8/replicate_${REP}
# vLLM forwards this seed to NumPy's legacy RNG, which requires uint32.
# Keep the eight replicate seeds distinct while staying below 2**32 - 1.
SEED=$((2026091000 + REP * 10000000))
test -f "${BRIDGES}/BRIDGE_BATCH_COMPLETE"
mkdir -p "${OUT}/logs"
pids=()
for shard_id in $(seq 0 7); do
  (
    set -euo pipefail
    export CUDA_VISIBLE_DEVICES=${shard_id}
    export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
    export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
    export VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_WORKER_MULTIPROC_METHOD=spawn
    export VLLM_PORT=$((22000 + (SLURM_JOB_ID % 1000) * 8 + shard_id))
    export XDG_CACHE_HOME=/tmp/t1k_sbridge_${SLURM_JOB_ID}_${shard_id}/xdg
    export VLLM_CACHE_ROOT=/tmp/t1k_sbridge_${SLURM_JOB_ID}_${shard_id}/vllm
    export TORCHINDUCTOR_CACHE_DIR=/tmp/t1k_sbridge_${SLURM_JOB_ID}_${shard_id}/torchinductor
    export TRITON_CACHE_DIR=/tmp/t1k_sbridge_${SLURM_JOB_ID}_${shard_id}/triton
    mkdir -p "${XDG_CACHE_HOME}" "${VLLM_CACHE_ROOT}" "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"
    "${PYTHON}" "${MODULE}/run_full_rollout_batch.py" --mode student_after_bridge \
      --bridge-manifest "${BRIDGES}/manifest.jsonl" --output-root "${OUT}" \
      --shard-id "${shard_id}" --num-shards 8 --model "${MODEL}" --tokenizer "${MODEL}" \
      --temperature 1.0 --top-p 1.0 --top-k -1 --seed "${SEED}" --first-step-logprobs 16 \
      --max-model-len 40960 --num-rollouts 8 --trace-batch-size 4 --gpu-memory-utilization 0.90
  ) >"${OUT}/logs/${SLURM_JOB_ID}_shard_${shard_id}.out" 2>"${OUT}/logs/${SLURM_JOB_ID}_shard_${shard_id}.err" &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
test "${failed}" -eq 0
selected=$(wc -l <"${BRIDGES}/manifest.jsonl")
touch "${OUT}/GPU_GENERATION_COMPLETE"
"${PYTHON}" "${MODULE}/verify_full_rollout_batch.py" --mode student_after_bridge \
  --bridge-manifest "${BRIDGES}/manifest.jsonl" --output-root "${OUT}" --num-shards 8 \
  --num-rollouts 8 --model "${MODEL}" --tokenizer "${MODEL}" --max-model-len 40960 \
  --temperature 1.0 --top-p 1.0 --top-k -1 --seed "${SEED}" --expected-count "${selected}"
