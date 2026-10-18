#!/usr/bin/env bash
#SBATCH --job-name=repair_tfull_marker
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/dataset/train/teacher_full_qwen3_14b_n8_seed20260826/logs/%j.repair.out
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/dataset/train/teacher_full_qwen3_14b_n8_seed20260826/logs/%j.repair.err

set -euo pipefail
ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
OUTPUT=${ROOT}/dataset/train/teacher_full_qwen3_14b_n8_seed20260826
PYTHON=/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python
VERIFIER=${ROOT}/ocrd_pipeline/full_rollout/verify_full_rollout_batch.py

test ! -f "${OUTPUT}/GPU_GENERATION_COMPLETE"
test ! -f "${OUTPUT}/FULL_ROLLOUT_BATCH_COMPLETE"
for shard_id in $(seq 0 7); do
  shard_name=$(printf '%02d' "${shard_id}")
  test -f "${OUTPUT}/shards/shard_${shard_name}/SHARD_COMPLETE"
done
count=$(find "${OUTPUT}/traces" -mindepth 2 -maxdepth 2 \
  -name teacher_full_n8.json -type f | wc -l)
test "${count}" -eq 13244
test -n "${AUDIT_JOB_ID:-}"
test "$(/soft/slurm/bin/sacct -j "${AUDIT_JOB_ID}" --format=State,ExitCode -n -P | head -n1)" = "COMPLETED|0:0"
ATTEST=${OUTPUT}/logs/audit_${AUDIT_JOB_ID}
test -s "${ATTEST}.ok"
test -s "${ATTEST}.sha256"
VERIFIER_SHA=$(sha256sum "${VERIFIER}" | cut -d' ' -f1)
grep -Fx "audit_job_id=${AUDIT_JOB_ID}" "${ATTEST}.ok"
grep -Fx "source_gpu_job=94080" "${ATTEST}.ok"
grep -Fx "verifier_sha256=${VERIFIER_SHA}" "${ATTEST}.ok"
grep -Fx "trace_count=13244" "${ATTEST}.ok"
if pgrep -af 'run_full_rollout_(batch.py|worker.sh)' | grep -F -- "${OUTPUT}"; then
  echo "active writer detected for ${OUTPUT}" >&2
  exit 1
fi
"${PYTHON}" "${VERIFIER}" \
  --mode teacher_from_anchor \
  --anchor-root "${ROOT}/ocrd_anchor_train25_qwen3_1_7b_nonthinking_seed20260811_transformers_v2/judge_remote_gpt56_luna_batch4" \
  --output-root "${OUTPUT}" \
  --num-shards 8 \
  --num-rollouts 8 \
  --model /storage/yukaichengLab/share/model/Qwen3-14B \
  --max-model-len 40960 \
  --expected-count 13244 \
  --check-only
find "${OUTPUT}/traces" -type f -name teacher_full_n8.json -print0 \
  | sort -z \
  | xargs -0 sha256sum > "${ATTEST}.repair.sha256"
cmp "${ATTEST}.sha256" "${ATTEST}.repair.sha256"
printf '%s\n' \
  "source_gpu_job=94080" \
  "readonly_audit_job=${AUDIT_JOB_ID:?AUDIT_JOB_ID is required}" \
  "repair_job=${SLURM_JOB_ID}" \
  "verified_trace_count=${count}" \
  "verifier_sha256=${VERIFIER_SHA}" \
  "content_hash_attestation=${ATTEST}.sha256" \
  "repaired_at=$(date --iso-8601=seconds)" \
  >"${OUTPUT}/GPU_GENERATION_COMPLETE.repair_provenance"
touch "${OUTPUT}/GPU_GENERATION_COMPLETE"
echo "TEACHER_GPU_GENERATION_MARKER_REPAIRED audit=${AUDIT_JOB_ID} traces=${count}"
