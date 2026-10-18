#!/bin/bash
#SBATCH --job-name=weak_to_8b_prefix_score
#SBATCH --partition=yukaichenglab
#SBATCH --account=yukaicheng
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --output=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/weak_to_8b_prefix_score.%j.log
#SBATCH --error=/storage/yukaichengLab/wangxin/test_eiloeo/opd_logs/weak_to_8b_prefix_score.%j.err
set -Eeuo pipefail
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
ROOT=/storage/yukaichengLab/wangxin/test_eiloeo
CODE=$ROOT/opd_scripts/weak_to_8b_prefix_handoff_20261005
OUT=$ROOT/opd_eval_prefix_handoff_weak_to_8b_20261005
PYTHON=/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python
mkdir -p "$OUT/judge"
cp /storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_ablation/main_greedy_k1_t2_20260921/judge/.relay_token "$OUT/judge/.relay_token"
cp "$CODE/system_prompt.txt" "$OUT/judge/system_prompt.txt"
rm -f "$OUT/judge/relay.stop" "$OUT/judge/relay.json"
cat > "$OUT/judge/EXTERNAL_SCORING_APPROVED.json" <<'JSON'
{"authorized_by":"user","instruction":"完成弱模型前缀接力实验并自动评分","scope":"two arms x five prefix percentages x eight matched benchmarks; 22540 math/reasoning grades via existing OpenLux gpt-4o-mini protocol","model":"gpt-4o-mini"}
JSON
"$PYTHON" "$CODE/relay.py" > "$OUT/judge/relay.log" 2>&1 &
RELAY_PID=$!
for _ in $(seq 1 60); do
  test -f "$OUT/judge/relay.json" && break
  sleep 1
done
test -f "$OUT/judge/relay.json"
"$PYTHON" "$CODE/score.py"
touch "$OUT/judge/relay.stop"
wait "$RELAY_PID" || true
