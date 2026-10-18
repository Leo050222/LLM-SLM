#!/bin/bash
#SBATCH --job-name=student_only_x8_score
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/student_only_x8_score.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/student_only_x8_score.%j.err

set -euo pipefail
source /storage/yukaichengLab/wangxin/miniconda3/etc/profile.d/conda.sh
conda activate verl-opd

ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
SCRIPT=${ROOT}/opd_scripts/score_deepmath_checkpoint_nonthinking_x8.py
OUTPUT_ROOT=${ROOT}/opd_eval_deepmath/ge6_test1000_student_only_nonthinking_x8_seed20260811
RUN=${OUTPUT_ROOT}/student_only
MODEL=/storage/yukaichengLab/share/model/Qwen3-1.7B

export PYTHONPATH=${ROOT}/opd_scripts:${ROOT}/verl
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

test -f "${OUTPUT_ROOT}/STUDENT_ONLY_GENERATION_COMPLETE"
test -f "${RUN}/GENERATION_COMPLETE"
if [[ -e "${OUTPUT_ROOT}/STUDENT_ONLY_SCORING_COMPLETE" ]]; then
  echo "Refusing to overwrite completed scoring: ${OUTPUT_ROOT}" >&2
  exit 3
fi
echo "STUDENT_ONLY_X8_SCORING_START job=${SLURM_JOB_ID} node=${SLURMD_NODENAME} cpus=${SLURM_CPUS_PER_TASK} time=$(date --iso-8601=seconds)"
python "${SCRIPT}" \
  --shard-root "${RUN}/shards" --output-dir "${RUN}/scored" \
  --model-label student_only --expected-model "${MODEL}" \
  --num-shards 4 --num-rollouts 8 --workers 64 --score-timeout 20

python - "${RUN}/scored" "${OUTPUT_ROOT}" <<'PY'
import json, sys
from pathlib import Path

scored = Path(sys.argv[1])
root = Path(sys.argv[2])
assert (scored / "SCORING_COMPLETE").is_file()
summary = json.loads((scored / "summary.json").read_text())
assert summary["status"] == "DEEPMATH_CHECKPOINT_NONTHINKING_X8_SCORING_COMPLETE"
assert summary["questions"] == 1000
assert summary["rollouts"] == 8000
assert summary["rollouts_per_question"] == 8
assert summary["thinking_open_tag"] == 0
assert summary["thinking_close_tag"] == 0
manifest = {"status": "STUDENT_ONLY_NONTHINKING_X8_SCORING_COMPLETE", "model": summary}
tmp = root / "scoring_manifest.json.tmp"
tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True))
tmp.replace(root / "scoring_manifest.json")
PY

touch "${OUTPUT_ROOT}/STUDENT_ONLY_SCORING_COMPLETE"
echo "STUDENT_ONLY_NONTHINKING_X8_SCORING_COMPLETE job=${SLURM_JOB_ID} time=$(date --iso-8601=seconds) output=${OUTPUT_ROOT}"
