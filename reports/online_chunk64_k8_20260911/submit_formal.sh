#!/bin/bash
# Run on login only AFTER the recorded smoke checks pass.
set -euo pipefail
export PATH=/soft/slurm/bin:$PATH
export OPD_CODE_ROOT=/storage/yukaichengLab/wangxin/test_eiloeo/opd_online_code/20260911_chunk64_k8
export OPD_PYTHON=/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python
export OPD_ONLINE_RUN=/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_deepmath/online_chunk64_k8_test1000_seed20260911
export OPD_JUDGE_CONFIG=/storage/yukaichengLab/wangxin/test_eiloeo/ocrd_pipeline/anchor_selector/config_gpt56_luna_openlux.json
export OPD_JUDGE_SECRETS=/storage/yukaichengLab/wangxin/test_eiloeo/ocrd_pipeline/anchor_selector/.secrets.openlux.env
export OPD_CONCURRENCY=64
SMOKE=/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_deepmath/online_chunk64_k8_smoke20260911
"$OPD_PYTHON" - "$SMOKE" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
assert json.loads((root/'live_contract.json').read_text())['status']=='LIVE_WORKER_CONTRACT_OK'
status=json.loads((root/'results/run_status.json').read_text())
assert status['completed']==8 and not status['failed']
assert json.loads((root/'smoke_audit.json').read_text())['status']=='REAL_JUDGE_SMOKE_AUDIT_OK'
assert json.loads((root/'results/summary.json').read_text())['questions']==8
assert json.loads((root/'judge_probe.json').read_text())['status']=='JUDGE_PROBE_OK'
assert len(json.loads((root/'long_prefix_benchmark.json').read_text())['results'])==8
rows=[json.loads(p.read_text()) for p in (root/'results/completed').glob('*.json')]
assert len(rows)==8
assert max(sum('bridge_token_ids' in e for e in row['events']) for row in rows)>=2
print('FORMAL_PREFLIGHT_SMOKE_PASSED')
PY
if [[ -e "$OPD_ONLINE_RUN/launch.json" || -e "$OPD_ONLINE_RUN/results/manifest.json" ]]; then
  echo 'Formal run already exists; inspect/resume instead of duplicating it' >&2
  exit 1
fi
mkdir -p "$OPD_ONLINE_RUN/logs"
cp "$SMOKE/prompts1000.jsonl" "$OPD_ONLINE_RUN/prompts1000.jsonl"
cp "$SMOKE/prompts1000.manifest.json" "$OPD_ONLINE_RUN/prompts1000.manifest.json"
# Slurm selects the GPUs. The smoke node is used only after that job exits.
gpu_job=$(sbatch --parsable --nodelist=gnho033 --time=24:00:00 \
  --output="$OPD_ONLINE_RUN/logs/%j.out" --error="$OPD_ONLINE_RUN/logs/%j.err" \
  "$OPD_CODE_ROOT/ocrd_pipeline/online_rollout/launch_workers.slurm")
printf '%s\n' "$gpu_job" > "$OPD_ONLINE_RUN/gpu_job.id"
nohup bash "$OPD_CODE_ROOT/ocrd_pipeline/online_rollout/launch_coordinator.sh" \
  > "$OPD_ONLINE_RUN/logs/coordinator.log" 2>&1 < /dev/null &
coordinator_pid=$!
printf '%s\n' "$coordinator_pid" > "$OPD_ONLINE_RUN/coordinator.pid"
score_job=$(sbatch --parsable --dependency="afterok:$gpu_job" \
  --output="$OPD_ONLINE_RUN/logs/%j_score.out" --error="$OPD_ONLINE_RUN/logs/%j_score.err" \
  "$OPD_CODE_ROOT/ocrd_pipeline/online_rollout/score.slurm")
"$OPD_PYTHON" - "$OPD_ONLINE_RUN" "$gpu_job" "$score_job" "$coordinator_pid" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
value=dict(gpu_job=sys.argv[2],score_job=sys.argv[3],coordinator_pid=int(sys.argv[4]),
           gpus=2,student_gpus=1,teacher_gpus=1,concurrency=64,questions=1000,
           chunk_tokens=64,teacher_tokens=8,gap_threshold=2.,rollouts_per_question=1,
           takeover_count_limit=None,resource_ceiling_nodes=2,resource_ceiling_h800=16)
(root/'launch.json').write_text(json.dumps(value,indent=2))
print(json.dumps(value))
PY
