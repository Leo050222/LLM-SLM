#!/usr/bin/env bash
#SBATCH --job-name=sbridge_horz_smoke
#SBATCH --partition=yukaichenglab
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=384G
#SBATCH --gres=gpu:H800:8
#SBATCH --exclusive
#SBATCH --exclude=gnho006,gnho009,gnho017,gnho018,gnho032,gnho034,gnho036,gnho038
#SBATCH --time=02:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/sbridge_horz_smoke.%A_%a.out
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/sbridge_horz_smoke.%A_%a.err

set -euo pipefail
ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
PYTHON=/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python
HORIZONS=(32 64 128)
H=${HORIZONS[${SLURM_ARRAY_TASK_ID}]}
PARENT=${ROOT}/opd_smoke/student_after_bridge_qwen3_1_7b_h${H}_n8_x8_seed20260826
REPLICATE=${PARENT}/replicate_00
SEED=20260826

[[ ! -e "${REPLICATE}/FULL_ROLLOUT_BATCH_COMPLETE" ]] || {
  echo "Refusing to overwrite completed smoke ${REPLICATE}" >&2
  exit 7
}
bash "${ROOT}/ocrd_pipeline/full_rollout/run_student_after_bridge_replicate_group.sh" \
  "${PARENT}" 0 0 1 "${H}"
"${PYTHON}" "${ROOT}/ocrd_pipeline/full_rollout/verify_full_rollout_batch.py" \
  --mode student_after_bridge \
  --bridge-manifest "${ROOT}/dataset/train/anchor_bridge_qwen3_14b_h256_n8/manifest.jsonl" \
  --output-root "${REPLICATE}" --num-shards 8 --max-records 1 \
  --num-rollouts 8 --bridge-token-limit "${H}" \
  --model /storage/yukaichengLab/share/model/Qwen3-1.7B \
  --max-model-len 40960 --seed "${SEED}" --expected-count 8
test -f "${REPLICATE}/FULL_ROLLOUT_BATCH_COMPLETE"
echo "STUDENT_BRIDGE_HORIZON_SMOKE_COMPLETE horizon=${H} job=${SLURM_JOB_ID}"
