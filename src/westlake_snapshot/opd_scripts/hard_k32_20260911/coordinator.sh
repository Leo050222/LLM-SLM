#!/bin/bash
set -Eeuo pipefail
R=/storage/yukaichengLab/wangxin/test_eiloeo
O=$R/opd_eval_hard/hard_k32_20260911
S=$R/opd_scripts/hard_k32_20260911
PY=/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
trap 'rc=$?; if ((rc)); then touch "$O/FAILED"; fi' EXIT
ready=0
for ((i=0;i<8640;i++)); do
 ready=1
 for ((k=0;k<8;k++)); do
  for role in student teacher; do test -f "$O/queue/replica_$k/$role/model.json" || ready=0; done
 done
 ((ready)) && break
 sleep 5
done
test "$ready" -eq 1
success=0
for attempt in 1 2 3; do
 if "$PY" "$S/resume_skip.py" --input "$O/prompts.jsonl" --output-dir "$O/online" --queue "$O/queue" --replicas 8 --tokenizer /storage/yukaichengLab/share/model/Qwen3-1.7B --judge-config "$R/ocrd_pipeline/anchor_selector/config_gpt56_luna_openlux.json" --secrets "$R/ocrd_pipeline/anchor_selector/.secrets.openlux.env" --expected-rows 1814 --concurrency 256 --chunk-tokens 64 --teacher-tokens 32 --max-tokens 40264 --seed 20260911; then success=1; break; fi
 echo "Retrying incomplete questions, attempt $attempt"
done
test "$success" -eq 1
"$PY" "$S/verify.py"
