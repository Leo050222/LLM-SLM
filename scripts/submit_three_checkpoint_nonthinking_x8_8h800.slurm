#!/bin/bash
#SBATCH --job-name=three_ckpt_nt_x8
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=512G
#SBATCH --gres=gpu:H800:8
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/three_ckpt_nt_x8.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/three_ckpt_nt_x8.%j.err

set -euo pipefail
source /storage/yukaichengLab/wangxin/miniconda3/etc/profile.d/conda.sh
conda activate verl-opd

ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
SCRIPT=${ROOT}/opd_scripts/run_deepmath_checkpoint_nonthinking_x8.py
MERGER=${ROOT}/opd_scripts/merge_deepmath_checkpoint_nonthinking_x8.py
FLAT_DATA=${ROOT}/opd_data/deepmath_ge6_opd_train15pct_seed20260811/eval_shards4/test_1000_flat.parquet
FULL_DATA=${ROOT}/opd_data/deepmath_ge6_opd_train15pct_seed20260811/test_1000.parquet
TOKENIZER=/storage/yukaichengLab/share/model/Qwen3-1.7B
OUTPUT_ROOT=${ROOT}/opd_eval_deepmath/ge6_test1000_three_ckpts_nonthinking_x8_seed20260811
LOG_ROOT=${ROOT}/opd_logs/three_ckpt_nt_x8.${SLURM_JOB_ID}.workers
EXPECTED_DATA_SHA=6ef8e66d181b7522bad2904ad1fcf20d8fc152af0c9d7eb05e07a9f2f0415ae0

LABELS=(
  traditional_opd_step133
  bridge_opd_step1656
  good_bridge_opd_step590
)
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

test -s "${SCRIPT}"
test -s "${MERGER}"
test -s "${FLAT_DATA}"
test -s "${FULL_DATA}"
test -s "${TOKENIZER}/tokenizer_config.json"
observed_data_sha=$(sha256sum "${FULL_DATA}" | awk '{print $1}')
test "${observed_data_sha}" = "${EXPECTED_DATA_SHA}"
for model in "${MODELS[@]}"; do
  test -s "${model}/config.json"
  test -s "${model}/model.safetensors.index.json"
done
if [[ -e "${OUTPUT_ROOT}/ALL_THREE_NONTHINKING_X8_COMPLETE" ]]; then
  echo "Refusing to overwrite completed experiment: ${OUTPUT_ROOT}" >&2
  exit 3
fi
mkdir -p "${OUTPUT_ROOT}/attempts" "${LOG_ROOT}"

echo "THREE_CHECKPOINT_NONTHINKING_X8_START job=${SLURM_JOB_ID} node=${SLURMD_NODENAME} time=$(date --iso-8601=seconds)"
/soft/slurm/bin/scontrol show job "${SLURM_JOB_ID}"
gpu_count=$(nvidia-smi -L | wc -l)
test "${gpu_count}" -eq 8
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -v 'NVIDIA H800' >/dev/null; then
  echo "Allocated node contains a non-H800 GPU" >&2
  exit 4
fi
other_jobs=$(/soft/slurm/bin/squeue -h -w "${SLURMD_NODENAME}" -o '%A' | sort -u | grep -vx "${SLURM_JOB_ID}" || true)
if [[ -n "${other_jobs}" ]]; then
  echo "Exclusive-node invariant failed: ${other_jobs}" >&2
  exit 5
fi
max_used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -nr | head -1)
test "${max_used_mib}" -lt 1024
echo "SINGLE_NODE_EXCLUSIVE_OK node=${SLURMD_NODENAME} gpu_count=${gpu_count} max_used_mib=${max_used_mib}"

for model_index in 0 1 2; do
  label=${LABELS[${model_index}]}
  model=${MODELS[${model_index}]}
  run=${OUTPUT_ROOT}/${label}
  if [[ -e "${run}/merged/NONTHINKING_X8_COMPLETE" ]]; then
    echo "STRICT_MODEL_RESULT_REUSED label=${label}"
    continue
  fi
  echo "MODEL_X8_START label=${label} model=${model} time=$(date --iso-8601=seconds)"
  pids=()
  for rank in $(seq 0 7); do
    (
      set -euo pipefail
      export CUDA_VISIBLE_DEVICES=${rank}
      export OPD_WORKER_RANK=${rank}
      export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12
      export XDG_CACHE_HOME=/tmp/xdg_three_ckpt_nt_x8_${SLURM_JOB_ID}_${rank}
      export VLLM_CACHE_ROOT=/tmp/vllm_three_ckpt_nt_x8_${SLURM_JOB_ID}_${rank}
      export TRITON_CACHE_DIR=/tmp/triton_three_ckpt_nt_x8_${SLURM_JOB_ID}_${rank}
      export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_three_ckpt_nt_x8_${SLURM_JOB_ID}_${rank}
      mkdir -p "${XDG_CACHE_HOME}" "${VLLM_CACHE_ROOT}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"
      python -c 'import torch; assert torch.cuda.is_available() and torch.cuda.device_count()==1; torch.manual_seed(20260811); x=torch.arange(8,device="cuda"); assert int(x.sum())==28; print("CUDA_WITNESS_OK",torch.cuda.get_device_name())'

      final=${run}/shards/shard_${rank}
      attempt=${OUTPUT_ROOT}/attempts/job_${SLURM_JOB_ID}_${label}_shard_${rank}
      if [[ -e "${final}" ]]; then
        echo "Refusing pre-existing unmerged shard: ${final}" >&2
        exit 6
      fi
      if [[ -e "${attempt}" ]]; then
        echo "Refusing pre-existing attempt: ${attempt}" >&2
        exit 7
      fi
      task=$((model_index * 8 + rank))
      seed_port=$((20000 + (SLURM_JOB_ID % 1000) * 32 + task))
      selected_port=""
      for offset in $(seq 0 127); do
        candidate=$((seed_port + offset * 32))
        if python -c 'import socket,sys; s=socket.socket(); s.bind(("127.0.0.1",int(sys.argv[1]))); s.close()' "${candidate}" 2>/dev/null; then
          selected_port=${candidate}
          break
        fi
      done
      test -n "${selected_port}"
      export VLLM_PORT=${selected_port}
      setsid python "${SCRIPT}" \
        --input "${FLAT_DATA}" --output-dir "${attempt}" \
        --model "${model}" --tokenizer "${TOKENIZER}" \
        --shard-index "${rank}" --num-shards 8 --expected-total-rows 1000 \
        --num-rollouts 8 --seed 20260811 --max-tokens 40264 \
        --max-model-len 40960 --score-timeout 20
      mkdir -p "$(dirname "${final}")"
      mv "${attempt}" "${final}"
      echo "MODEL_X8_SHARD_OK label=${label} rank=${rank} time=$(date --iso-8601=seconds)"
    ) >"${LOG_ROOT}/${label}.rank_${rank}.log" 2>"${LOG_ROOT}/${label}.rank_${rank}.err" &
    pids+=("$!")
  done

  failed=0
  for rank in $(seq 0 7); do
    if ! wait "${pids[$rank]}"; then
      echo "MODEL_X8_WORKER_FAILED label=${label} rank=${rank}" >&2
      failed=1
    fi
  done
  test "${failed}" -eq 0
  python "${MERGER}" \
    --shard-root "${run}/shards" --output-dir "${run}/merged" \
    --full-data "${FULL_DATA}" --expected-data-sha256 "${EXPECTED_DATA_SHA}" \
    --expected-model "${model}" --expected-tokenizer "${TOKENIZER}" \
    --model-label "${label}" --num-shards 8 --num-rollouts 8 \
    --seed 20260811 --max-tokens 40264 --max-model-len 40960
  echo "MODEL_X8_COMPLETE label=${label} time=$(date --iso-8601=seconds)"
  sleep 30
done

python -c 'import json,sys; from pathlib import Path; root=Path(sys.argv[1]); labels=sys.argv[2:]; summaries=[]
for label in labels:
 p=root/label/"merged"; assert (p/"NONTHINKING_X8_COMPLETE").is_file(); s=json.load((p/"summary.json").open()); assert s["status"]=="DEEPMATH_CHECKPOINT_NATIVE_NONTHINKING_X8_COMPLETE" and s["questions"]==1000 and s["rollouts"]==8000; summaries.append(s)
manifest={"status":"ALL_THREE_CHECKPOINTS_NONTHINKING_X8_COMPLETE","models":summaries}; tmp=root/"manifest.json.tmp"; tmp.write_text(json.dumps(manifest,indent=2,sort_keys=True)); tmp.replace(root/"manifest.json")' "${OUTPUT_ROOT}" "${LABELS[@]}"
touch "${OUTPUT_ROOT}/ALL_THREE_NONTHINKING_X8_COMPLETE"
echo "ALL_THREE_CHECKPOINTS_NONTHINKING_X8_COMPLETE job=${SLURM_JOB_ID} time=$(date --iso-8601=seconds) output=${OUTPUT_ROOT}"
