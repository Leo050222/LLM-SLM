#!/bin/bash
#SBATCH --job-name=opd_dm25_nt_rkl
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=384G
#SBATCH --gres=gpu:H800:8
#SBATCH --exclusive
#SBATCH --exclude=gnho009,gnho018,gnho038
#SBATCH --time=1-00:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/opd_dm25_nt_rkl.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/opd_dm25_nt_rkl.%j.err

set -euo pipefail
source /storage/yukaichengLab/wangxin/miniconda3/etc/profile.d/conda.sh
conda activate verl-opd

export PYTHONPATH=/storage/yukaichengLab/wangxin/test_eiloeo/verl
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 NCCL_DEBUG=WARN
export CUDA_DEVICE_ORDER=PCI_BUS_ID VLLM_ENABLE_V1_MULTIPROCESSING=0 OMP_NUM_THREADS=8

python -c 'import torch; assert torch.cuda.device_count()==8, torch.cuda.device_count(); [(torch.ones(32,32,device=f"cuda:{i}") @ torch.ones(32,32,device=f"cuda:{i}")).sum().item() for i in range(8)]; print("CUDA_PREFLIGHT_OK", torch.cuda.device_count())'

SMOKE_RUN=/storage/yukaichengLab/wangxin/test_eiloeo/opd_smoke/deepmath_ge6_train25pct_nonthinking_student_top16_rkl_8h800_single_node_twostep
SMOKE_CKPT=/storage/yukaichengLab/wangxin/test_eiloeo/opd_smoke_checkpoints/deepmath_ge6_train25pct_nonthinking_student_top16_rkl_8h800_single_node_twostep
python /storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/verify_deepmath_top16_smoke.py \
  "${SMOKE_RUN}" "${SMOKE_CKPT}" --expected-steps 2 \
  --expected-global-batch 64 --expected-checkpoints 1,2 \
  --expected-student-world-size 4 --expected-max-prompt-tokens 699 \
  --expected-thinking false --verify-hf-load

export OPD_SCRIPT=/storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/run_deepmath_full_vocab_reverse_kl_train.py
export OPD_RUN_DIR=/storage/yukaichengLab/wangxin/test_eiloeo/opd_runs/qwen3_1_7b_from_14b_deepmath_ge6_train25pct_nonthinking_student_top16_rkl_8h800_single_node
export OPD_CHECKPOINT_ROOT=/storage/yukaichengLab/wangxin/test_eiloeo/opd_checkpoints/qwen3_1_7b_from_14b_deepmath_ge6_train25pct_nonthinking_student_top16_rkl_8h800_single_node
export OPD_TRAIN_FILE=/storage/yukaichengLab/wangxin/test_eiloeo/opd_data/deepmath_ge6_opd_train25pct_perdifficulty_seed20260823/train.parquet
export OPD_TRAIN_SHA256=fdcb58732d22f5ab27d111f07c8746836d983c5d15d1d800970154ea1f3cfc91
export OPD_DATA_STATUS=DEEPMATH_GE6_TRAIN25PCT_PER_DIFFICULTY_OK
export OPD_EXPECTED_DATA_TRAIN_STEPS=222
export OPD_TRAIN_ROWS=14208 OPD_SEED=20260811
export OPD_GLOBAL_BATCH_SIZE=64 OPD_STUDENT_WORLD_SIZE=4
export OPD_TOTAL_STEPS=222 OPD_SAVE_STEPS=44,89,133,178,222 OPD_LOSS_SUPPORT_TOPK=16
export OPD_ENABLE_THINKING=0 OPD_MAX_PROMPT_TOKENS=699
export OPD_MAX_MODEL_LEN=40960 OPD_MAX_RESPONSE_TOKENS=40264
export OPD_HF_MAX_SHARD_SIZE=2GB

bash /storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/run_deepmath_top16_8h800_single_node_launcher.sh
