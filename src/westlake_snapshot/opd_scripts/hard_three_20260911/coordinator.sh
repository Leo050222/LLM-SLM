#!/bin/bash
set -Eeuo pipefail
R=/storage/yukaichengLab/wangxin/test_eiloeo
O=$R/opd_eval_hard/hard_three_20260911
S=$R/opd_scripts/hard_three_20260911
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
"$PY" "$R/opd_online_code/20260911_chunk64_k8_2nodes/ocrd_pipeline/online_rollout/run.py" --input "$O/prompts.jsonl" --output-dir "$O/online" --queue "$O/queue" --replicas 8 --tokenizer /storage/yukaichengLab/share/model/Qwen3-1.7B --judge-config "$R/ocrd_pipeline/anchor_selector/config_gpt56_luna_openlux.json" --secrets "$R/ocrd_pipeline/anchor_selector/.secrets.openlux.env" --expected-rows 1814 --concurrency 256 --chunk-tokens 64 --teacher-tokens 8 --max-tokens 40264 --seed 20260911
"$PY" "$S/generate.py" student
"$PY" "$S/generate.py" teacher
"$PY" "$S/generate.py" verify
