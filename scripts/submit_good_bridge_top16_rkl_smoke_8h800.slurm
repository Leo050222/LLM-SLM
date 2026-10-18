#!/usr/bin/env bash
#SBATCH --job-name=goodbr_top16_smoke
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=384G
#SBATCH --gres=gpu:H800:8
#SBATCH --exclusive
#SBATCH --exclude=gnho006,gnho009,gnho017,gnho018,gnho036,gnho038
#SBATCH --time=04:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/good_bridge_top16_smoke.%j.out
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/good_bridge_top16_smoke.%j.err

set -euo pipefail
source /storage/yukaichengLab/wangxin/miniconda3/etc/profile.d/conda.sh
conda activate verl-opd
ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
export PYTHONPATH=${ROOT}/verl
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 NCCL_DEBUG=WARN
export CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=8

test "$(nvidia-smi -L | wc -l)" -eq 8
max_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk 'BEGIN{m=0}{if($1>m)m=$1}END{print m}')
test "${max_used}" -lt 500
python -c 'import torch; assert torch.cuda.device_count()==8; torch.manual_seed(20260827); values=[(torch.arange(16,device=f"cuda:{i}")**2).sum().item() for i in range(8)]; assert values==[1240]*8; print("GOOD_BRIDGE_TOP16_CUDA_WITNESS_OK", values)'

export OPD_SCRIPT=${ROOT}/opd_scripts/run_bridge_top16_reverse_kl_train.py
export OPD_RUN_DIR=${ROOT}/opd_smoke/bridge_h256_good37724_student_top16_rkl_seed20260827_smoke
export OPD_CHECKPOINT_ROOT=${ROOT}/opd_smoke_checkpoints/bridge_h256_good37724_student_top16_rkl_seed20260827_smoke
export OPD_BRIDGE_ROOT=${ROOT}/ocrd_anchor_train25_qwen3_1_7b_nonthinking_seed20260811_transformers_v2/bridge_qwen3_14b_h256_n8_seed20260822
export OPD_BRIDGE_SELECTION=${ROOT}/dataset/train/student_good_bridge_selection_qwen3_1_7b_h256_n8_seed20260827/selection.json
export OPD_SEED=20260827 OPD_GLOBAL_BATCH_SIZE=64 OPD_STUDENT_WORLD_SIZE=4
export OPD_TOTAL_STEPS=2 OPD_SAVE_STEPS=1,2
export OPD_MAX_MODEL_LEN=40960 OPD_HF_MAX_SHARD_SIZE=2GB

bash ${ROOT}/opd_scripts/run_bridge_top16_8h800_single_node_launcher.sh
python ${ROOT}/opd_scripts/verify_bridge_top16_reverse_kl_train.py \
  "${OPD_RUN_DIR}" "${OPD_CHECKPOINT_ROOT}" \
  --expected-steps 2 --expected-checkpoints 1,2 \
  --expected-covered-bridges 128 --expected-train-rows 37724 \
  --expected-selection "${OPD_BRIDGE_SELECTION}" --verify-hf-load
echo GOOD_BRIDGE_TOP16_RKL_SMOKE_COMPLETE
