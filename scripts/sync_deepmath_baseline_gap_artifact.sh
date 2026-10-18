#!/bin/bash
set -u

REMOTE=/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_deepmath/ge6_test1000_qwen3_1_7b_baseline_seed20260811/sampled_token_gap_visualization_compact.tar
LOCAL=/Users/leolee/Documents/OPD/reports/deepmath_baseline_sample_gap_compact.tar
REPORTS=/Users/leolee/Documents/OPD/reports
TARGET=${REPORTS}/deepmath_baseline_sample_gap
ATTEMPT=0

while true; do
  ATTEMPT=$((ATTEMPT + 1))
  echo "SYNC_ATTEMPT=${ATTEMPT} START=$(date -Iseconds)"
  if rsync -az --partial --append -e "ssh -o ConnectTimeout=60 -p 10025" \
    "wangxin@172.16.78.35:${REMOTE}" "${LOCAL}"; then
    break
  fi
  if rsync -az --partial --append -e "ssh -o ConnectTimeout=60 -p 10024" \
    "wangxin@172.16.78.35:${REMOTE}" "${LOCAL}"; then
    break
  fi
  echo "SYNC_RETRY_AFTER_60S=$(date -Iseconds)"
  sleep 60
done

tar -tf "${LOCAL}" >/dev/null
tar -xf "${LOCAL}" -C "${REPORTS}"
rsync -a "${REPORTS}/sampled_token_gap_visualization_compact/" "${TARGET}/"
touch "${TARGET}/SYNC_COMPLETE"
echo "SYNC_COMPLETE=$(date -Iseconds) FILES=$(find "${TARGET}/data" -name 'case_*.js' | wc -l)"
