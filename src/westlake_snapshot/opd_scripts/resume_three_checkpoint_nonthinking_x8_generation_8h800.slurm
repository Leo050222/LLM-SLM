#!/bin/bash
#SBATCH --job-name=three_ckpt_x8_gen
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=512G
#SBATCH --gres=gpu:H800:8
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/three_ckpt_x8_gen.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/three_ckpt_x8_gen.%j.err

set -euo pipefail
source /storage/yukaichengLab/wangxin/miniconda3/etc/profile.d/conda.sh
conda activate verl-opd

ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
SCRIPT=${ROOT}/opd_scripts/run_deepmath_checkpoint_nonthinking_x8.py
DATA=${ROOT}/opd_data/deepmath_ge6_opd_train15pct_seed20260811/eval_shards4/test_1000_flat.parquet
TOKENIZER=/storage/yukaichengLab/share/model/Qwen3-1.7B
OUTPUT_ROOT=${ROOT}/opd_eval_deepmath/ge6_test1000_three_ckpts_nonthinking_x8_seed20260811
LOG_ROOT=${ROOT}/opd_logs/three_ckpt_x8_gen.${SLURM_JOB_ID}.workers

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
mkdir -p "${OUTPUT_ROOT}/attempts" "${LOG_ROOT}"

echo "THREE_CHECKPOINT_X8_GENERATION_START job=${SLURM_JOB_ID} node=${SLURMD_NODENAME} time=$(date --iso-8601=seconds)"
test "$(nvidia-smi -L | wc -l)" -eq 8

for model_index in 0 1 2; do
  label=${LABELS[${model_index}]}
  model=${MODELS[${model_index}]}
  run=${OUTPUT_ROOT}/${label}
  mkdir -p "${run}/shards"
  pids=()
  for rank in $(seq 0 7); do
    (
      set -euo pipefail
      final=${run}/shards/shard_${rank}
      if [[ -s "${final}/rollouts.jsonl" && -s "${final}/run_config.json" && -f "${final}/SHARD_COMPLETE" ]]; then
        test "$(wc -l < "${final}/rollouts.jsonl")" -eq 1000
        python -c 'import json,sys; c=json.load(open(sys.argv[1])); assert c["student"]==sys.argv[2] and int(c["shard_index"])==int(sys.argv[3]) and int(c["num_rollouts"])==1000' "${final}/run_config.json" "${model}" "${rank}"
        echo "GENERATION_SHARD_REUSED label=${label} rank=${rank}"
        exit 0
      fi
      if [[ -e "${final}" ]]; then
        echo "Refusing incomplete pre-existing final shard: ${final}" >&2
        exit 6
      fi
      export CUDA_VISIBLE_DEVICES=${rank} OPD_WORKER_RANK=${rank}
      export OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10
      export XDG_CACHE_HOME=/tmp/xdg_three_ckpt_x8_gen_${SLURM_JOB_ID}_${rank}
      export VLLM_CACHE_ROOT=/tmp/vllm_three_ckpt_x8_gen_${SLURM_JOB_ID}_${rank}
      export TRITON_CACHE_DIR=/tmp/triton_three_ckpt_x8_gen_${SLURM_JOB_ID}_${rank}
      export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_three_ckpt_x8_gen_${SLURM_JOB_ID}_${rank}
      mkdir -p "${XDG_CACHE_HOME}" "${VLLM_CACHE_ROOT}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"
      attempt=${OUTPUT_ROOT}/attempts/job_${SLURM_JOB_ID}_${label}_shard_${rank}
      test ! -e "${attempt}"
      export VLLM_PORT=$((21000 + (SLURM_JOB_ID % 500) * 32 + model_index * 8 + rank))
      setsid python "${SCRIPT}" --generate-only \
        --input "${DATA}" --output-dir "${attempt}" \
        --model "${model}" --tokenizer "${TOKENIZER}" \
        --shard-index "${rank}" --num-shards 8 --expected-total-rows 1000 \
        --num-rollouts 8 --seed 20260811 --max-tokens 40264 --max-model-len 40960
      mv "${attempt}" "${final}"
      echo "GENERATION_SHARD_OK label=${label} rank=${rank}"
    ) >"${LOG_ROOT}/${label}.rank_${rank}.log" 2>"${LOG_ROOT}/${label}.rank_${rank}.err" &
    pids+=("$!")
  done
  failed=0
  for rank in $(seq 0 7); do
    wait "${pids[$rank]}" || failed=1
  done
  test "${failed}" -eq 0
  for rank in $(seq 0 7); do
    test "$(wc -l < "${run}/shards/shard_${rank}/rollouts.jsonl")" -eq 1000
    test -f "${run}/shards/shard_${rank}/SHARD_COMPLETE"
  done
  touch "${run}/GENERATION_COMPLETE"
  echo "MODEL_GENERATION_COMPLETE label=${label} time=$(date --iso-8601=seconds)"
  sleep 30
done

touch "${OUTPUT_ROOT}/ALL_THREE_GENERATION_COMPLETE"
echo "ALL_THREE_CHECKPOINTS_X8_GENERATION_COMPLETE job=${SLURM_JOB_ID} time=$(date --iso-8601=seconds) output=${OUTPUT_ROOT}"
