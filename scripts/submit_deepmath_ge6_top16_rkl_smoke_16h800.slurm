#!/bin/bash
#SBATCH --job-name=opd_dm_top16_smoke
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=384G
#SBATCH --gres=gpu:H800:8
#SBATCH --exclude=gnho006,gnho008,gnho011,gnho016
#SBATCH --time=04:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/opd_dm_top16_smoke.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/opd_dm_top16_smoke.%j.err

set -euo pipefail
source /storage/yukaichengLab/wangxin/miniconda3/etc/profile.d/conda.sh
conda activate verl-opd

export PYTHONPATH=/storage/yukaichengLab/wangxin/test_eiloeo/verl
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 NCCL_DEBUG=WARN
export CUDA_DEVICE_ORDER=PCI_BUS_ID VLLM_ENABLE_V1_MULTIPROCESSING=0 OMP_NUM_THREADS=8

export OPD_SCRIPT=/storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/run_deepmath_full_vocab_reverse_kl_train.py
export OPD_RUN_DIR=/storage/yukaichengLab/wangxin/test_eiloeo/opd_smoke/deepmath_ge6_train15pct_student_top16_rkl_16h800
export OPD_CHECKPOINT_ROOT=/storage/yukaichengLab/wangxin/test_eiloeo/opd_smoke_checkpoints/deepmath_ge6_train15pct_student_top16_rkl_16h800
export OPD_TRAIN_FILE=/storage/yukaichengLab/wangxin/test_eiloeo/opd_data/deepmath_ge6_opd_train15pct_seed20260811/train.parquet
export OPD_TRAIN_SHA256=4e3b711079515c1748269ef89fd818fc7c895df6229a1264ee0c5235bcedd758
export OPD_DATA_STATUS=DEEPMATH_GE6_TRAIN15PCT_OK
export OPD_EXPECTED_DATA_TRAIN_STEPS=133
export OPD_TRAIN_ROWS=8512 OPD_SEED=20260811
export OPD_GLOBAL_BATCH_SIZE=16 OPD_STUDENT_WORLD_SIZE=8
export OPD_TOTAL_STEPS=1 OPD_SAVE_STEPS=1 OPD_LOSS_SUPPORT_TOPK=16

bash /storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/run_deepmath_top16_16h800_launcher.sh
python /storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/verify_deepmath_top16_smoke.py \
  "${OPD_RUN_DIR}" "${OPD_CHECKPOINT_ROOT}"
