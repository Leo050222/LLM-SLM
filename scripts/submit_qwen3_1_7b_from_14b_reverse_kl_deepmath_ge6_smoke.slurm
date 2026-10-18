#!/bin/bash
#SBATCH --job-name=opd_dm_ge6_smoke
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=384G
#SBATCH --gres=gpu:H800:8
#SBATCH --time=03:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/opd_dm_ge6_smoke.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/opd_dm_ge6_smoke.%j.err

set -euo pipefail

source /storage/yukaichengLab/wangxin/miniconda3/etc/profile.d/conda.sh
conda activate verl-opd

export PYTHONPATH=/storage/yukaichengLab/wangxin/test_eiloeo/verl
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export NCCL_DEBUG=WARN
export VLLM_PORT=$((20000 + SLURM_JOB_ID % 20000))

DATA_DIR=/storage/yukaichengLab/wangxin/test_eiloeo/opd_data/deepmath_ge6_opd
CHECKPOINT_DIR=/storage/yukaichengLab/wangxin/test_eiloeo/opd_checkpoints/qwen3_1_7b_from_14b_reverse_kl_deepmath_ge6_smoke
RUN_LOG=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/opd_dm_ge6_smoke_run.${SLURM_JOB_ID}.log

test "$(python -c 'import json; print(json.load(open("'"${DATA_DIR}"'/manifest.json"))["status"])')" = "DEEPMATH_GE6_OPD_DATA_OK"
mkdir -p "${CHECKPOINT_DIR}"
cd /storage/yukaichengLab/wangxin/test_eiloeo/verl

TRAIN_FILE="${DATA_DIR}/smoke_train.parquet" \
VAL_FILE="${DATA_DIR}/test_1000.parquet" \
CHECKPOINT_DIR="${CHECKPOINT_DIR}" \
TRAIN_BATCH_SIZE=4 \
PPO_MINI_BATCH_SIZE=4 \
TOTAL_EPOCHS=1 \
SAVE_FREQ=1 \
bash /storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/run_qwen3_1_7b_from_14b_reverse_kl_deepmath_ge6.sh 2>&1 | tee "${RUN_LOG}"

test -f "${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt"
test "$(tr -d '[:space:]' < "${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt")" = "1"
test -d "${CHECKPOINT_DIR}/global_step_1/actor/huggingface"
find "${CHECKPOINT_DIR}/global_step_1/actor" -type f -name 'model_world_size_*_rank_*.pt' | grep -q .
find "${CHECKPOINT_DIR}/global_step_1/actor" -type f -name 'optim_world_size_*_rank_*.pt' | grep -q .
find "${CHECKPOINT_DIR}/global_step_1/actor" -type f -name 'extra_state_world_size_*_rank_*.pt' | grep -q .
grep -q 'distillation/loss' "${RUN_LOG}"
echo "DEEPMATH_GE6_REVERSE_KL_SMOKE_OK job=${SLURM_JOB_ID}"
