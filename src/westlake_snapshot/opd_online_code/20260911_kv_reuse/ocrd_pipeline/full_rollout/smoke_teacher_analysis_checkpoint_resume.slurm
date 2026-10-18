#!/usr/bin/env bash
#SBATCH --job-name=smoke_score_resume
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/dataset/train/teacher_full_qwen3_14b_n8_seed20260826/logs/%j.analysis_resume_smoke.out
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/dataset/train/teacher_full_qwen3_14b_n8_seed20260826/logs/%j.analysis_resume_smoke.err

set -euo pipefail
ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
ROLLOUT=${ROOT}/dataset/train/teacher_full_qwen3_14b_n8_seed20260826_smoke4
DATA=${ROOT}/dataset/train/original_deepmath_ge6_train25pct/train_unique.parquet
OUTPUT=${ROLLOUT}/analysis_checkpoint_resume_smoke_${SLURM_JOB_ID}
PYTHON=/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python
SCORER=${ROOT}/ocrd_pipeline/full_rollout/analyze_teacher_rollout_shard.py

test -f "${ROLLOUT}/FULL_ROLLOUT_BATCH_COMPLETE"
set +e
"${PYTHON}" "${SCORER}" \
  --rollout-root "${ROLLOUT}" \
  --dataset "${DATA}" \
  --output-root "${OUTPUT}" \
  --shard-index 0 \
  --num-shards 1 \
  --num-rollouts 8 \
  --score-timeout 20 \
  --fail-after-traces-for-test 2 \
  >"${OUTPUT}.first.out" 2>"${OUTPUT}.first.err"
first_rc=$?
set -e
if [[ "${first_rc}" -eq 0 ]]; then
  echo "Expected intentional first-pass failure" >&2
  exit 30
fi
test "$(find "${OUTPUT}/shards/shard_000/trace_checkpoints" -type f -name 'trace_*.json' | wc -l)" -eq 2

"${PYTHON}" "${SCORER}" \
  --rollout-root "${ROLLOUT}" \
  --dataset "${DATA}" \
  --output-root "${OUTPUT}" \
  --shard-index 0 \
  --num-shards 1 \
  --num-rollouts 8 \
  --score-timeout 20 \
  >"${OUTPUT}.resume.out" 2>"${OUTPUT}.resume.err"

test -f "${OUTPUT}/shards/shard_000/SCORE_SHARD_COMPLETE"
grep -q 'cached_traces=2' "${OUTPUT}.resume.out"
"${PYTHON}" - "${OUTPUT}/shards/shard_000/summary.json" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
assert summary["status"] == "TEACHER_ROLLOUT_ANALYSIS_SHARD_COMPLETE"
assert summary["trace_count"] == 4
assert summary["rollout_count"] == 32
assert summary["cached_trace_count"] == 2
assert summary["scored_trace_count_now"] == 2
print("TEACHER_ANALYSIS_CHECKPOINT_RESUME_SMOKE_OK", json.dumps(summary, sort_keys=True))
PY
