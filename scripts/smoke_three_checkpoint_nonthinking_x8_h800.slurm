#!/bin/bash
#SBATCH --job-name=three_ckpt_x8_smoke
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:H800:1
#SBATCH --time=01:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/three_ckpt_x8_smoke.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/three_ckpt_x8_smoke.%j.err

set -euo pipefail
source /storage/yukaichengLab/wangxin/miniconda3/etc/profile.d/conda.sh
conda activate verl-opd

ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
SCRIPT=${ROOT}/opd_scripts/run_deepmath_checkpoint_nonthinking_x8.py
INPUT=${ROOT}/opd_data/deepmath_ge6_opd_train15pct_seed20260811/eval_shards4/test_1000_flat.parquet
TOKENIZER=/storage/yukaichengLab/share/model/Qwen3-1.7B
SMOKE_ROOT=${ROOT}/opd_smoke/three_ckpts_nonthinking_x8_job${SLURM_JOB_ID}
LABELS=(traditional_opd_step133 bridge_opd_step1656 good_bridge_opd_step590)
MODELS=(
  ${ROOT}/opd_checkpoints/qwen3_1_7b_from_14b_deepmath_ge6_train15pct_student_top16_rkl_16h800/global_step_133/actor/huggingface
  ${ROOT}/opd_checkpoints/qwen3_1_7b_bridge_h256_n8_student_top16_rkl_seed20260827/global_step_1656/actor/huggingface
  ${ROOT}/opd_checkpoints/qwen3_1_7b_bridge_h256_good37724_student_top16_rkl_seed20260827/global_step_590/actor/huggingface
)

export PYTHONPATH=${ROOT}/opd_scripts:${ROOT}/verl
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12
export XDG_CACHE_HOME=/tmp/xdg_three_ckpt_x8_smoke_${SLURM_JOB_ID}
export VLLM_CACHE_ROOT=/tmp/vllm_three_ckpt_x8_smoke_${SLURM_JOB_ID}
export TRITON_CACHE_DIR=/tmp/triton_three_ckpt_x8_smoke_${SLURM_JOB_ID}
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_three_ckpt_x8_smoke_${SLURM_JOB_ID}
mkdir -p "${XDG_CACHE_HOME}" "${VLLM_CACHE_ROOT}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"

python -c 'import torch; assert torch.cuda.is_available() and torch.cuda.device_count()==1; torch.manual_seed(20260811); x=torch.randn(8,8,device="cuda"); y=x@x; assert y.isfinite().all(); print("SEEDED_KERNEL_WITNESS_OK",tuple(y.shape),torch.cuda.get_device_name())'
for index in 0 1 2; do
  label=${LABELS[${index}]}
  model=${MODELS[${index}]}
  output=${SMOKE_ROOT}/${label}
  test -s "${model}/config.json"
  python "${SCRIPT}" \
    --input "${INPUT}" --output-dir "${output}" \
    --model "${model}" --tokenizer "${TOKENIZER}" \
    --shard-index 0 --num-shards 1000 --expected-total-rows 1000 \
    --num-rollouts 8 --seed 20260811 --max-tokens 40264 \
    --max-model-len 40960 --score-timeout 20
  python -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); c=json.load((p/"run_config.json").open()); rows=[json.loads(x) for x in (p/"rollouts.jsonl").open()]; assert c["thinking_mode"] is False and c["sampling"]["n"]==8 and c["num_prompts"]==1 and c["num_rollouts"]==8; assert [r["replicate_index"] for r in rows]==list(range(8)); assert all("<think>" not in r["response"] and "</think>" not in r["response"] for r in rows); print("MODEL_NONTHINKING_X8_SMOKE_OK",sys.argv[2])' "${output}" "${label}"
  sleep 10
done
touch "${SMOKE_ROOT}/ALL_THREE_SMOKE_COMPLETE"
echo "ALL_THREE_CHECKPOINTS_NONTHINKING_X8_SMOKE_COMPLETE job=${SLURM_JOB_ID} output=${SMOKE_ROOT}"
