#!/bin/bash
#SBATCH --job-name=three_ckpt_x8_score
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/three_ckpt_x8_score.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/three_ckpt_x8_score.%j.err

set -euo pipefail
source /storage/yukaichengLab/wangxin/miniconda3/etc/profile.d/conda.sh
conda activate verl-opd

ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
SCRIPT=${ROOT}/opd_scripts/score_deepmath_checkpoint_nonthinking_x8.py
OUTPUT_ROOT=${ROOT}/opd_eval_deepmath/ge6_test1000_three_ckpts_nonthinking_x8_seed20260811
LABELS=(traditional_opd_step133 bridge_opd_step1656 good_bridge_opd_step590)
MODELS=(
  ${ROOT}/opd_checkpoints/qwen3_1_7b_from_14b_deepmath_ge6_train15pct_student_top16_rkl_16h800/global_step_133/actor/huggingface
  ${ROOT}/opd_checkpoints/qwen3_1_7b_bridge_h256_n8_student_top16_rkl_seed20260827/global_step_1656/actor/huggingface
  ${ROOT}/opd_checkpoints/qwen3_1_7b_bridge_h256_good37724_student_top16_rkl_seed20260827/global_step_590/actor/huggingface
)

export PYTHONPATH=${ROOT}/opd_scripts:${ROOT}/verl
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
test -f "${OUTPUT_ROOT}/ALL_THREE_GENERATION_COMPLETE"
echo "THREE_CHECKPOINT_X8_SCORING_START job=${SLURM_JOB_ID} node=${SLURMD_NODENAME} cpus=${SLURM_CPUS_PER_TASK} time=$(date --iso-8601=seconds)"

for index in 0 1 2; do
  label=${LABELS[${index}]}
  model=${MODELS[${index}]}
  run=${OUTPUT_ROOT}/${label}
  test -f "${run}/GENERATION_COMPLETE"
  python "${SCRIPT}" \
    --shard-root "${run}/shards" --output-dir "${run}/scored_v2" \
    --model-label "${label}" --expected-model "${model}" \
    --num-shards 8 --num-rollouts 8 --workers 64 --score-timeout 20
  echo "MODEL_SCORING_COMPLETE label=${label} time=$(date --iso-8601=seconds)"
done

python -c 'import json,sys; from pathlib import Path; root=Path(sys.argv[1]); labels=sys.argv[2:]; summaries=[]
for label in labels:
 p=root/label/"scored_v2"; assert (p/"SCORING_COMPLETE").is_file(); s=json.load((p/"summary.json").open()); assert s["rollouts"]==8000; summaries.append(s)
manifest={"status":"ALL_THREE_CHECKPOINTS_NONTHINKING_X8_SCORING_COMPLETE","models":summaries}; tmp=root/"scoring_manifest.json.tmp"; tmp.write_text(json.dumps(manifest,indent=2,sort_keys=True)); tmp.replace(root/"scoring_manifest.json")' "${OUTPUT_ROOT}" "${LABELS[@]}"
touch "${OUTPUT_ROOT}/ALL_THREE_SCORING_COMPLETE"
echo "ALL_THREE_CHECKPOINTS_X8_SCORING_COMPLETE job=${SLURM_JOB_ID} time=$(date --iso-8601=seconds)"
