#!/bin/bash
#SBATCH --job-name=dm_max_smoke
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --nodelist=gnho020
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:H800:1
#SBATCH --array=0-2
#SBATCH --time=04:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/dm_max_smoke.%A_%a.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/dm_max_smoke.%A_%a.err

set -euo pipefail
source /storage/yukaichengLab/wangxin/miniconda3/etc/profile.d/conda.sh
conda activate verl-opd

ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
SCRIPT=${ROOT}/opd_scripts/run_opd_logit_diagnostic.py
DATA=${ROOT}/opd_data/deepmath_103k/sample_1000.parquet
OUTPUT=${ROOT}/opd_eval_deepmath/maxlen_smoke_${SLURM_ARRAY_JOB_ID}
TASK=${SLURM_ARRAY_TASK_ID}
MODELS=(Qwen3-1.7B Qwen3-4B Qwen3-14B)
MODEL=${MODELS[${TASK}]}
out=${OUTPUT}/${MODEL}

export PYTHONPATH=${ROOT}/verl
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=8 VLLM_WORKER_MULTIPROC_METHOD=spawn PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_PORT=$((20000 + (SLURM_ARRAY_JOB_ID % 1000) * 16 + TASK))
export TRITON_CACHE_DIR=/tmp/triton_dm_max_smoke_${SLURM_ARRAY_JOB_ID}_${TASK}
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_dm_max_smoke_${SLURM_ARRAY_JOB_ID}_${TASK}
mkdir -p "${out}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"

echo "ARRAY_JOB=${SLURM_ARRAY_JOB_ID} TASK=${TASK} MODEL=${MODEL} VLLM_PORT=${VLLM_PORT} NODE=${SLURMD_NODENAME} START=$(date --iso-8601=seconds)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

common=(
  --student /storage/yukaichengLab/share/model/${MODEL}
  --teacher /storage/yukaichengLab/share/model/Qwen3-14B
  --dataset-name deepmath_103k --data "${DATA}" --prompt-style deepmath --scorer math
  --score-timeout 20 --seed 20260809 --max-tokens 40512 --max-model-len 40960 --topk 128 --limit 1
)
python "${SCRIPT}" generate --output-dir "${out}" "${common[@]}"
echo "PHASE_GENERATE_DONE=$(date --iso-8601=seconds)"
if [[ "${TASK}" -lt 2 ]]; then
  python "${SCRIPT}" teacher --output-dir "${out}" "${common[@]}"
  echo "PHASE_TEACHER_DONE=$(date --iso-8601=seconds)"
  python "${SCRIPT}" student --output-dir "${out}" "${common[@]}"
  echo "PHASE_STUDENT_DONE=$(date --iso-8601=seconds)"
  python "${SCRIPT}" summarize --output-dir "${out}" "${common[@]}"
fi

python -c "import json,pathlib; p=pathlib.Path('${out}'); c=json.loads((p/'run_config.json').read_text()); r=[json.loads(x) for x in (p/'rollouts.jsonl').read_text().splitlines()]; assert len(r)==1 and c['sampling']['max_tokens']==40512 and c['max_model_len']==40960 and c['thinking_mode']; print('MAXLEN_SMOKE_OK', '${MODEL}', r[0]['response_tokens'], r[0]['finish_reason'])"
