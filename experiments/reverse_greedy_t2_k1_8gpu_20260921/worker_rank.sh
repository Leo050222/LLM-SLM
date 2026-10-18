#!/bin/bash
set -euo pipefail
rank=${SLURM_PROCID:?}
replica=$((rank / 2))
if (( rank % 2 == 0 )); then
 role=student
 model=/storage/yukaichengLab/share/model/Qwen3-14B
else
 role=teacher
 model=/storage/yukaichengLab/share/model/Qwen3-1.7B
fi
queue="$OPD_ONLINE_RUN/queue/replica_$replica"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=8
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export XDG_CACHE_HOME="$OPD_ONLINE_RUN/cache/rank_$rank/xdg"
export TORCHINDUCTOR_CACHE_DIR="$OPD_ONLINE_RUN/cache/rank_$rank/torchinductor"
export TRITON_CACHE_DIR="$OPD_ONLINE_RUN/cache/rank_$rank/triton"
mkdir -p "$queue" "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
if [[ -f "$queue/STOP" ]]; then echo 'Queue is stopped'; exit 1; fi
printf 'WORKER_BINDING rank=%s replica=%s role=%s node=%s visible=%s\n' "$rank" "$replica" "$role" "$SLURMD_NODENAME" "${CUDA_VISIBLE_DEVICES:-unset}"
exec "$OPD_PYTHON" "$SCALE_CODE/worker.py" \
 --queue "$queue" --role "$role" --model "$model"
