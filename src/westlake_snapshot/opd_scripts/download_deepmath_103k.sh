#!/bin/bash
set -euo pipefail

REVISION=5cf055d1fe3d7a2eb19719ac020211469736ae44
BASE_URL="https://hf-mirror.com/datasets/zwhe99/DeepMath-103K/resolve/${REVISION}"
OUTPUT=${1:-/storage/yukaichengLab/wangxin/test_eiloeo/opd_data/deepmath_103k/source}

mkdir -p "${OUTPUT}/data"

download() {
  local relative_path=$1
  local target="${OUTPUT}/${relative_path}"
  local partial="${target}.part"
  if [[ -s "${target}" ]]; then
    echo "REUSE ${target}"
    return
  fi
  mkdir -p "$(dirname "${target}")"
  curl -L \
    --fail \
    --retry 30 \
    --retry-delay 5 \
    --retry-all-errors \
    --connect-timeout 30 \
    --continue-at - \
    --output "${partial}" \
    "${BASE_URL}/${relative_path}"
  mv "${partial}" "${target}"
  echo "DOWNLOADED ${target}"
}

download README.md
for shard in $(seq 0 9); do
  relative_path=$(printf "data/train-%05d-of-00010.parquet" "${shard}")
  download "${relative_path}"
done

(
  cd "${OUTPUT}"
  sha256sum data/train-*.parquet > SHA256SUMS
)
cat > "${OUTPUT}/SOURCE.json.tmp" <<EOF
{
  "dataset": "zwhe99/DeepMath-103K",
  "revision": "${REVISION}",
  "mirror": "https://hf-mirror.com",
  "num_source_shards": 10
}
EOF
mv "${OUTPUT}/SOURCE.json.tmp" "${OUTPUT}/SOURCE.json"
touch "${OUTPUT}/DOWNLOAD_COMPLETE"
echo "DEEPMATH_DOWNLOAD_COMPLETE ${OUTPUT}"
