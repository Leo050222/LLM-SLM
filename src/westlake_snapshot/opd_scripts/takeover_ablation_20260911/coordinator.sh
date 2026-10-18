#!/bin/bash
set -Eeuo pipefail
R=/storage/yukaichengLab/wangxin/test_eiloeo
O=$R/opd_eval_hard/takeover_ablation_20260911
S=$R/opd_scripts/takeover_ablation_20260911
PY=/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
trap 'rc=$?; if ((rc)); then touch "$O/FAILED"; fi' EXIT
ready=0
for ((i=0;i<8640;i++)); do
 ready=1
 for ((k=0;k<4;k++)); do
  for role in student teacher; do test -f "$O/queue/replica_$k/$role/model.json" || ready=0; done
 done
 ((ready)) && break
 sleep 5
done
test "$ready" -eq 1
for mode in luna_cap8 nojudge_cap8 luna_allow random_allow; do
 ok=0
 for attempt in 1 2 3; do
  if "$PY" "$S/run.py" --mode "$mode" --input "$O/prompts.jsonl" --output-dir "$O/$mode" --queue "$O/queue" --replicas 4 --tokenizer /storage/yukaichengLab/share/model/Qwen3-1.7B --judge-config "$R/ocrd_pipeline/anchor_selector/config_gpt56_luna_openlux.json" --secrets "$R/ocrd_pipeline/anchor_selector/.secrets.openlux.env" --expected-rows 2254 --concurrency 128 --chunk-tokens 64 --teacher-tokens 8 --gap-threshold 2 --max-tokens 40264 --seed 20260911; then ok=1; break; fi
 done
 test "$ok" -eq 1
 "$PY" "$S/verify.py" "$mode"
done
"$PY" "$S/verify.py" all
