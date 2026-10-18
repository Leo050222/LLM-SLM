#!/bin/bash
#SBATCH --job-name=t1k_sbridge_score
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=256G
#SBATCH --array=0-7%3
#SBATCH --time=08:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_deepmath/ge6_test1000_anchor_bridge_h256_n8x8_seed20260910/logs/score_%A_%a.out
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_deepmath/ge6_test1000_anchor_bridge_h256_n8x8_seed20260910/logs/score_%A_%a.err

set -euo pipefail
ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
EXP=${ROOT}/opd_eval_deepmath/ge6_test1000_anchor_bridge_h256_n8x8_seed20260910
MODULE=${ROOT}/ocrd_pipeline/full_rollout
PYTHON=/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python
DATA=${ROOT}/opd_data/deepmath_ge6_opd_train15pct_seed20260811/test_1000.parquet
REP=${SLURM_ARRAY_TASK_ID}
ROLLOUT=${EXP}/student_after_bridge_n8x8/replicate_${REP}
OUT=${ROLLOUT}/analysis
NUM_SHARDS=32
test -f "${ROLLOUT}/FULL_ROLLOUT_BATCH_COMPLETE"
mkdir -p "${OUT}/logs"
pids=()
for shard_id in $(seq 0 $((NUM_SHARDS - 1))); do
  "${PYTHON}" "${MODULE}/analyze_teacher_rollout_shard.py" --rollout-kind student_after_bridge \
    --rollout-root "${ROLLOUT}" --dataset "${DATA}" --output-root "${OUT}" \
    --shard-index "${shard_id}" --num-shards "${NUM_SHARDS}" --num-rollouts 8 \
    --expected-dataset-rows 1000 --score-timeout 20 \
    >"${OUT}/logs/shard_${shard_id}.out" 2>"${OUT}/logs/shard_${shard_id}.err" &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
test "${failed}" -eq 0
selected=$(wc -l <"${ROLLOUT}/manifest.jsonl")
"${PYTHON}" "${MODULE}/merge_teacher_rollout_analysis.py" --rollout-kind student_after_bridge \
  --rollout-root "${ROLLOUT}" --analysis-root "${OUT}" --num-shards "${NUM_SHARDS}" \
  --expected-traces "${selected}" --num-rollouts 8
