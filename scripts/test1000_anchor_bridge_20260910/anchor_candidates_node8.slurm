#!/bin/bash
#SBATCH --job-name=t1k_anchor_scores
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --mem=1000G
#SBATCH --gres=gpu:H800:8
#SBATCH --exclusive
#SBATCH --exclude=gnho009,gnho018,gnho036
#SBATCH --time=03:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_deepmath/ge6_test1000_anchor_bridge_h256_n8x8_seed20260910/logs/anchor_%j.out
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_deepmath/ge6_test1000_anchor_bridge_h256_n8x8_seed20260910/logs/anchor_%j.err

set -euo pipefail
ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
EXP=${ROOT}/opd_eval_deepmath/ge6_test1000_anchor_bridge_h256_n8x8_seed20260910
SOURCE=${ROOT}/opd_eval_deepmath/ge6_test1000_native_nonthinking_seed20260811/qwen3_1_7b/merged/rollouts.jsonl
SHARDS=${EXP}/source
OUT=${EXP}/anchor_candidates
MODULE=${ROOT}/ocrd_pipeline/anchor_selector
PYTHON=/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python
PREP=${ROOT}/opd_scripts/test1000_anchor_bridge_20260910/prepare_anchor_shards.py

mkdir -p "${EXP}/logs" "${OUT}/shards"
"${PYTHON}" "${PREP}" --source "${SOURCE}" --output-root "${SHARDS}" \
  --expected-source-sha256 aa5e4c48f905b1bb8da284b506c1fa6f5cec678506a508b3ef48b3bac589a5a3 \
  --expected-rows 1000 --num-shards 8

echo "TEST1000_ANCHOR_GPU_START job=${SLURM_JOB_ID} node=${SLURMD_NODENAME}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
pids=()
for shard_id in $(seq 0 7); do
  (
    set -euo pipefail
    export CUDA_VISIBLE_DEVICES=${shard_id}
    export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
    export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
    export XDG_CACHE_HOME=/tmp/t1k_anchor_${SLURM_JOB_ID}_${shard_id}/xdg
    export TORCHINDUCTOR_CACHE_DIR=/tmp/t1k_anchor_${SLURM_JOB_ID}_${shard_id}/torchinductor
    export TRITON_CACHE_DIR=/tmp/t1k_anchor_${SLURM_JOB_ID}_${shard_id}/triton
    mkdir -p "${XDG_CACHE_HOME}" "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"
    shard=$(printf '%02d' "${shard_id}")
    trace=${SHARDS}/shards/shard_${shard}/rollouts.jsonl
    output=${OUT}/shards/shard_${shard}
    mkdir -p "${output}"
    "${PYTHON}" -c 'import torch; assert torch.cuda.is_available() and torch.cuda.device_count()==1; x=torch.arange(8,device="cuda"); assert int(x.sum())==28; print("CUDA_WITNESS_OK",torch.cuda.get_device_name())'
    "${PYTHON}" "${MODULE}/run_anchor_shard.py" teacher-transformers --trace-shard "${trace}" --output-dir "${output}" --resume --attention-implementation sdpa
    "${PYTHON}" "${MODULE}/run_anchor_shard.py" student-candidates --trace-shard "${trace}" --output-dir "${output}" --resume --attention-implementation sdpa
    touch "${output}/SHARD_COMPLETE"
  ) >"${EXP}/logs/anchor_${SLURM_JOB_ID}_shard_${shard_id}.out" 2>"${EXP}/logs/anchor_${SLURM_JOB_ID}_shard_${shard_id}.err" &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
test "${failed}" -eq 0
test "$(find "${OUT}/shards" -name SHARD_COMPLETE -type f | wc -l)" -eq 8
touch "${OUT}/GPU_SCORING_COMPLETE"
echo "TEST1000_ANCHOR_GPU_COMPLETE job=${SLURM_JOB_ID}"
