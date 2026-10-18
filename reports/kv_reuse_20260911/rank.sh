#!/bin/bash
set -euo pipefail
rank=$SLURM_PROCID
mode=cached
extra=()
if ((rank>=2)); then mode=reference; extra=(--disable-prefix-caching); fi
role=student
model=Qwen3-1.7B
if ((rank%2)); then role=teacher; model=Qwen3-14B; fi
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export XDG_CACHE_HOME="$OPD_KV_RUN/cache/$rank"
exec "$OPD_PYTHON" "$OPD_CODE_ROOT/ocrd_pipeline/online_rollout/worker.py" --queue "$OPD_KV_RUN/$mode" --role "$role" --model "/storage/yukaichengLab/share/model/$model" "${extra[@]}"
