#!/usr/bin/env bash
#SBATCH --job-name=audit_tfull_n8
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/dataset/train/teacher_full_qwen3_14b_n8_seed20260826/logs/%j.audit.out
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/dataset/train/teacher_full_qwen3_14b_n8_seed20260826/logs/%j.audit.err

set -euo pipefail
ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
OUTPUT=${ROOT}/dataset/train/teacher_full_qwen3_14b_n8_seed20260826
PYTHON=/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python
VERIFIER=${ROOT}/ocrd_pipeline/full_rollout/verify_full_rollout_batch.py
ATTEST=${OUTPUT}/logs/audit_${SLURM_JOB_ID}

test ! -f "${OUTPUT}/GPU_GENERATION_COMPLETE"
test ! -f "${OUTPUT}/FULL_ROLLOUT_BATCH_COMPLETE"
test "$(/soft/slurm/bin/sacct -j 94080 --format=State,ExitCode -n -P | head -n1)" = "COMPLETED|0:0"
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
VERIFIER_SHA=$(sha256sum "${VERIFIER}" | cut -d' ' -f1)
find "${OUTPUT}/traces" -type f -name teacher_full_n8.json -print0 \
  | sort -z \
  | xargs -0 sha256sum > "${ATTEST}.sha256.tmp"
test "$(wc -l < "${ATTEST}.sha256.tmp")" -eq 13244
mv "${ATTEST}.sha256.tmp" "${ATTEST}.sha256"
printf 'audit_job_id=%s\nsource_gpu_job=94080\nverifier_sha256=%s\ntrace_count=13244\n' \
  "${SLURM_JOB_ID}" "${VERIFIER_SHA}" > "${ATTEST}.ok.tmp"
mv "${ATTEST}.ok.tmp" "${ATTEST}.ok"
echo TEACHER_FULL_READONLY_AUDIT_OK
