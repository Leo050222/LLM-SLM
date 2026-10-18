#!/bin/bash
#SBATCH --job-name=opd_dm_ge6_rkl
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=384G
#SBATCH --gres=gpu:H800:8
#SBATCH --time=7-00:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/opd_dm_ge6_rkl.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/opd_dm_ge6_rkl.%j.err

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
CHECKPOINT_DIR=/storage/yukaichengLab/wangxin/test_eiloeo/opd_checkpoints/qwen3_1_7b_from_14b_reverse_kl_deepmath_ge6

python - <<'PY'
import json
from pathlib import Path
manifest = json.loads(Path("/storage/yukaichengLab/wangxin/test_eiloeo/opd_data/deepmath_ge6_opd/manifest.json").read_text())
assert manifest["status"] == "DEEPMATH_GE6_OPD_DATA_OK"
assert manifest["filtered_unique_rows"] == 57630
assert manifest["train_unique_rows"] == 56630
assert manifest["train_total_rows"] == 56640
assert manifest["train_steps"] == 885
assert manifest["test_rows"] == 1000
assert manifest["prompt_tokens"]["max"] == 695
print("DEEPMATH_GE6_MANIFEST_VERIFIED", manifest["files"])
PY

mkdir -p "${CHECKPOINT_DIR}"
cd /storage/yukaichengLab/wangxin/test_eiloeo/verl

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start time: $(date --iso-8601=seconds)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

bash /storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/run_qwen3_1_7b_from_14b_reverse_kl_deepmath_ge6.sh

echo "End time: $(date --iso-8601=seconds)"
