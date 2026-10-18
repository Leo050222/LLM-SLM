#!/bin/bash
#SBATCH --job-name=prefix_parallel_gen
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=288G
#SBATCH --gres=gpu:H800:3
#SBATCH --time=02:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/prefix_parallel.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/prefix_parallel.%j.err
set -euo pipefail
source /storage/yukaichengLab/wangxin/miniconda3/etc/profile.d/conda.sh
conda activate verl-opd
ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
export PYTHONPATH=${ROOT}/opd_scripts:${ROOT}/verl
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
IFS=',' read -r -a devices <<< "${CUDA_VISIBLE_DEVICES}"
test "${#devices[@]}" -eq 3
lengths=(64 128 256)
pids=()
for rank in 0 1 2; do
 (
 export CUDA_VISIBLE_DEVICES=${devices[$rank]}
 export VLLM_CACHE_ROOT=/tmp/vllm_prefix_${SLURM_JOB_ID}_${rank}
 export TRITON_CACHE_DIR=/tmp/triton_prefix_${SLURM_JOB_ID}_${rank}
 nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
 setsid python "${ROOT}/opd_scripts/teacher_prefix_handoff_20260909/generate.py" --length "${lengths[$rank]}" --output "${ROOT}/opd_eval_deepmath/teacher_prefix_handoff_20260909/parallel/h${lengths[$rank]}"
 ) >"${ROOT}/opd_logs/prefix_parallel.${SLURM_JOB_ID}.h${lengths[$rank]}.log" 2>&1 &
 pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
test "$failed" -eq 0
echo ALL_GENERATION_WORKERS_COMPLETE
