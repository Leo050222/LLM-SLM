#!/bin/bash
#SBATCH --job-name=deepmath17_smoke
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --gres=gpu:H100:1
#SBATCH --time=01:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/deepmath17_smoke.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/deepmath17_smoke.%j.err

set -euo pipefail

source /storage/yukaichengLab/wangxin/miniconda3/etc/profile.d/conda.sh
conda activate verl-opd

ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
SCRIPT=${ROOT}/opd_scripts/run_opd_logit_diagnostic.py
DATA=${ROOT}/opd_data/deepmath_103k/sample_1000.parquet
OUTPUT=${ROOT}/opd_eval_deepmath/smoke_1_7b_${SLURM_JOB_ID}
STUDENT=/storage/yukaichengLab/share/model/Qwen3-1.7B
TEACHER=/storage/yukaichengLab/share/model/Qwen3-14B

export PYTHONPATH=${ROOT}/verl
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=8
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton_deepmath17_smoke_${SLURM_JOB_ID}
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_deepmath17_smoke_${SLURM_JOB_ID}

mkdir -p "${OUTPUT}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"
echo "JOB_ID=${SLURM_JOB_ID} NODE=${SLURMD_NODENAME} START=$(date --iso-8601=seconds)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

common=(
  --student "${STUDENT}"
  --teacher "${TEACHER}"
  --dataset-name deepmath_103k
  --data "${DATA}"
  --prompt-style deepmath
  --scorer math
  --score-timeout 20
  --seed 20260809
  --max-tokens 4096
  --max-model-len 8192
  --topk 128
)

python "${SCRIPT}" generate --output-dir "${OUTPUT}" --limit 2 "${common[@]}"
echo "PHASE_GENERATE_DONE=$(date --iso-8601=seconds)"
python "${SCRIPT}" teacher --output-dir "${OUTPUT}" "${common[@]}"
echo "PHASE_TEACHER_DONE=$(date --iso-8601=seconds)"
python "${SCRIPT}" student --output-dir "${OUTPUT}" "${common[@]}"
echo "PHASE_STUDENT_DONE=$(date --iso-8601=seconds)"
python "${SCRIPT}" summarize --output-dir "${OUTPUT}" "${common[@]}"

python - "${OUTPUT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = [json.loads(line) for line in (root / "rollouts.jsonl").read_text().splitlines()]
config = json.loads((root / "run_config.json").read_text())
assert len(rows) == 2
assert all("<think>" in row["response"] for row in rows)
assert config["student"] == "/storage/yukaichengLab/share/model/Qwen3-1.7B"
assert config["teacher"] == "/storage/yukaichengLab/share/model/Qwen3-14B"
assert config["sampling"] == {
    "max_tokens": 4096,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": -1,
    "do_sample": True,
    "n": 1,
    "seed": 20260809,
}
assert config["student_parameters_require_grad"] is False
assert len(list((root / "teacher_topk").glob("*.npz"))) == 2
assert len(list((root / "token_shards").glob("*.parquet"))) == 2
print("DEEPMATH_17B_SMOKE_OK")
PY

echo "DEEPMATH_17B_SMOKE_COMPLETE=$(date --iso-8601=seconds)"
