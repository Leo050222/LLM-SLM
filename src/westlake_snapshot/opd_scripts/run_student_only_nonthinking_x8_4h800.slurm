#!/bin/bash
#SBATCH --job-name=student_only_nt_x8
#SBATCH --partition=yukaichenglab
#SBATCH --nodelist=gnho041
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=384G
#SBATCH --gres=gpu:H800:4
#SBATCH --time=04:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/student_only_nt_x8.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/student_only_nt_x8.%j.err

set -euo pipefail
source /storage/yukaichengLab/wangxin/miniconda3/etc/profile.d/conda.sh
conda activate verl-opd

ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
SCRIPT=${ROOT}/opd_scripts/run_deepmath_checkpoint_nonthinking_x8.py
FLAT_DATA=${ROOT}/opd_data/deepmath_ge6_opd_train15pct_seed20260811/eval_shards4/test_1000_flat.parquet
FULL_DATA=${ROOT}/opd_data/deepmath_ge6_opd_train15pct_seed20260811/test_1000.parquet
MODEL=/storage/yukaichengLab/share/model/Qwen3-1.7B
TOKENIZER=/storage/yukaichengLab/share/model/Qwen3-1.7B
OUTPUT_ROOT=${ROOT}/opd_eval_deepmath/ge6_test1000_student_only_nonthinking_x8_seed20260811
RUN=${OUTPUT_ROOT}/student_only
LOG_ROOT=${ROOT}/opd_logs/student_only_nt_x8.${SLURM_JOB_ID}.workers
EXPECTED_DATA_SHA=6ef8e66d181b7522bad2904ad1fcf20d8fc152af0c9d7eb05e07a9f2f0415ae0

export PYTHONPATH=${ROOT}/opd_scripts:${ROOT}/verl
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

test -s "${SCRIPT}"
test -s "${FLAT_DATA}"
test -s "${FULL_DATA}"
test -s "${MODEL}/config.json"
test -s "${MODEL}/model.safetensors.index.json"
observed_data_sha=$(sha256sum "${FULL_DATA}" | awk '{print $1}')
test "${observed_data_sha}" = "${EXPECTED_DATA_SHA}"
if [[ -e "${OUTPUT_ROOT}/STUDENT_ONLY_GENERATION_COMPLETE" ]]; then
  echo "Refusing to overwrite completed experiment: ${OUTPUT_ROOT}" >&2
  exit 3
fi
mkdir -p "${OUTPUT_ROOT}/attempts" "${LOG_ROOT}"

gpu_spec=${CUDA_VISIBLE_DEVICES:-${SLURM_JOB_GPUS:-}}
test -n "${gpu_spec}"
echo "STUDENT_ONLY_NONTHINKING_X8_START job=${SLURM_JOB_ID} node=${SLURMD_NODENAME} visible=${gpu_spec} time=$(date --iso-8601=seconds)"
/soft/slurm/bin/scontrol show job "${SLURM_JOB_ID}"
IFS=',' read -r -a gpu_ids <<< "${gpu_spec}"
test "${#gpu_ids[@]}" -eq 4

pids=()
for rank in $(seq 0 3); do
  (
    set -euo pipefail
    export CUDA_VISIBLE_DEVICES=${gpu_ids[$rank]}
    export OPD_WORKER_RANK=${rank}
    export OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10
    export XDG_CACHE_HOME=/tmp/xdg_student_only_nt_x8_${SLURM_JOB_ID}_${rank}
    export VLLM_CACHE_ROOT=/tmp/vllm_student_only_nt_x8_${SLURM_JOB_ID}_${rank}
    export TRITON_CACHE_DIR=/tmp/triton_student_only_nt_x8_${SLURM_JOB_ID}_${rank}
    export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_student_only_nt_x8_${SLURM_JOB_ID}_${rank}
    mkdir -p "${XDG_CACHE_HOME}" "${VLLM_CACHE_ROOT}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"
    python -c 'import torch; assert torch.cuda.is_available() and torch.cuda.device_count()==1; torch.manual_seed(20260811); x=torch.arange(8,device="cuda"); assert int(x.sum())==28; print("CUDA_WITNESS_OK",torch.cuda.get_device_name())'

    final=${RUN}/shards/shard_${rank}
    attempt=${OUTPUT_ROOT}/attempts/job_${SLURM_JOB_ID}_shard_${rank}
    if [[ -e "${final}" ]]; then
      echo "Refusing pre-existing shard: ${final}" >&2
      exit 6
    fi
    if [[ -e "${attempt}" ]]; then
      echo "Refusing pre-existing attempt: ${attempt}" >&2
      exit 7
    fi
    seed_port=$((20000 + (SLURM_JOB_ID % 1000) * 16 + rank))
    selected_port=""
    for offset in $(seq 0 255); do
      candidate=$((seed_port + offset * 16))
      if python -c 'import socket,sys; s=socket.socket(); s.bind(("127.0.0.1",int(sys.argv[1]))); s.close()' "${candidate}" 2>/dev/null; then
        selected_port=${candidate}
        break
      fi
    done
    test -n "${selected_port}"
    export VLLM_PORT=${selected_port}
    setsid python "${SCRIPT}" \
      --input "${FLAT_DATA}" --output-dir "${attempt}" \
      --model "${MODEL}" --tokenizer "${TOKENIZER}" \
      --shard-index "${rank}" --num-shards 4 --expected-total-rows 1000 \
      --num-rollouts 8 --seed 20260811 --max-tokens 40264 \
      --max-model-len 40960 --score-timeout 20 --generate-only
    mkdir -p "$(dirname "${final}")"
    mv "${attempt}" "${final}"
    echo "STUDENT_ONLY_X8_SHARD_OK rank=${rank} time=$(date --iso-8601=seconds)"
  ) >"${LOG_ROOT}/rank_${rank}.log" 2>"${LOG_ROOT}/rank_${rank}.err" &
  pids+=("$!")
done

failed=0
for rank in $(seq 0 3); do
  if ! wait "${pids[$rank]}"; then
    echo "STUDENT_ONLY_X8_WORKER_FAILED rank=${rank}" >&2
    failed=1
  fi
done
test "${failed}" -eq 0

python - "${RUN}/shards" "${MODEL}" <<'PY'
import hashlib, json, sys
from pathlib import Path

root = Path(sys.argv[1])
model = sys.argv[2]
keys = set()
for shard_index in range(4):
    shard = root / f"shard_{shard_index}"
    assert (shard / "SHARD_COMPLETE").is_file()
    cfg = json.loads((shard / "run_config.json").read_text())
    assert cfg["student"] == model
    assert cfg["shard_index"] == shard_index
    path = shard / "rollouts.jsonl"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == cfg["output_sha256"]
    rows = [json.loads(line) for line in path.open() if line.strip()]
    assert len(rows) == 2000
    for row in rows:
        key = (int(row["dataset_index"]), int(row["replicate_index"]))
        assert int(row["dataset_index"]) % 4 == shard_index
        assert key not in keys
        keys.add(key)
assert keys == {(i, j) for i in range(1000) for j in range(8)}
print("STUDENT_ONLY_X8_GENERATION_VERIFIED", len(keys))
PY

touch "${RUN}/GENERATION_COMPLETE"
touch "${OUTPUT_ROOT}/STUDENT_ONLY_GENERATION_COMPLETE"
echo "STUDENT_ONLY_NONTHINKING_X8_GENERATION_COMPLETE job=${SLURM_JOB_ID} time=$(date --iso-8601=seconds) output=${OUTPUT_ROOT}"
