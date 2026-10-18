#!/bin/bash
#SBATCH --job-name=deepmath_smoke
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:H100:2
#SBATCH --time=01:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/deepmath_smoke.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/deepmath_smoke.%j.err

set -euo pipefail

source /storage/yukaichengLab/wangxin/miniconda3/etc/profile.d/conda.sh
conda activate verl-opd

ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
SCRIPT=${ROOT}/opd_scripts/run_opd_logit_diagnostic.py
DATA=${ROOT}/opd_data/deepmath_103k/sample_1000.parquet
OUTPUT=${ROOT}/opd_eval_deepmath/smoke_${SLURM_JOB_ID}
export OUTPUT

export PYTHONPATH=${ROOT}/verl
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=8
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton_deepmath_smoke_${SLURM_JOB_ID}
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_deepmath_smoke_${SLURM_JOB_ID}

mkdir -p "${OUTPUT}/qwen3_4b" "${OUTPUT}/qwen3_14b" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"

echo "JOB_ID=${SLURM_JOB_ID} NODE=${SLURMD_NODENAME} START=$(date --iso-8601=seconds)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

common=(
  --dataset-name deepmath_103k
  --data "${DATA}"
  --prompt-style deepmath
  --scorer math
  --score-timeout 20
  --seed 20260809
  --max-tokens 4096
  --max-model-len 8192
  --topk 128
  --limit 2
)

(
  export CUDA_VISIBLE_DEVICES=0
  out=${OUTPUT}/qwen3_4b
  python "${SCRIPT}" generate --output-dir "${out}" "${common[@]}"
  python "${SCRIPT}" teacher --output-dir "${out}" --topk 128 --seed 20260809 --max-tokens 4096 --max-model-len 8192
  python "${SCRIPT}" student --output-dir "${out}" --topk 128 --seed 20260809 --max-tokens 4096 --max-model-len 8192
  python "${SCRIPT}" summarize --output-dir "${out}" --topk 128 --seed 20260809 --max-tokens 4096 --max-model-len 8192
) > "${OUTPUT}/qwen3_4b.log" 2>&1 &
student_pid=$!

(
  export CUDA_VISIBLE_DEVICES=1
  python "${SCRIPT}" generate \
    --output-dir "${OUTPUT}/qwen3_14b" \
    --student /storage/yukaichengLab/share/model/Qwen3-14B \
    --teacher /storage/yukaichengLab/share/model/Qwen3-14B \
    "${common[@]}"
) > "${OUTPUT}/qwen3_14b.log" 2>&1 &
teacher_pid=$!

wait "${student_pid}"
wait "${teacher_pid}"

python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["OUTPUT"])
for name in ("qwen3_4b", "qwen3_14b"):
    rows = [json.loads(line) for line in (root / name / "rollouts.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert all("<think>" in row["response"] for row in rows)
    config = json.loads((root / name / "run_config.json").read_text())
    assert config["sampling"]["temperature"] == 1.0
    assert config["sampling"]["top_p"] == 1.0
    assert config["sampling"]["top_k"] == -1
    assert config["sampling"]["max_tokens"] == 4096
assert len(list((root / "qwen3_4b" / "teacher_topk").glob("*.npz"))) == 2
assert len(list((root / "qwen3_4b" / "token_shards").glob("*.parquet"))) == 2
print("DEEPMATH_SMOKE_OK")
PY

echo "DEEPMATH_SMOKE_COMPLETE=$(date --iso-8601=seconds)"
