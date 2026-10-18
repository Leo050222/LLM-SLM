#!/bin/bash
set -euo pipefail
export PATH=/soft/slurm/bin:$PATH
export OPD_CODE_ROOT=/storage/yukaichengLab/wangxin/test_eiloeo/opd_online_code/20260911_chunk64_k8_2nodes
export OPD_PYTHON=/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python
export OPD_ONLINE_RUN=/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_deepmath/online_chunk64_k8_test1000_2nodes_seed20260911
export OPD_JUDGE_CONFIG=/storage/yukaichengLab/wangxin/test_eiloeo/ocrd_pipeline/anchor_selector/config_gpt56_luna_openlux.json
export OPD_JUDGE_SECRETS=/storage/yukaichengLab/wangxin/test_eiloeo/ocrd_pipeline/anchor_selector/.secrets.openlux.env
if [[ -e "$OPD_ONLINE_RUN/launch.json" || -e "$OPD_ONLINE_RUN/gpu_job.id" ]]; then
 echo 'Run already submitted; inspect rather than duplicate' >&2; exit 1
fi
test -f "$OPD_ONLINE_RUN/results/migration.json"
if [[ -n "$(squeue -h -j 102149 -o %T)" ]]; then
 echo 'Old GPU job still active; refusing to exceed two-node ceiling' >&2; exit 1
fi
mkdir -p "$OPD_ONLINE_RUN/logs"
gpu_job=$(sbatch --parsable --nodelist=gnho011,gnho034 \
 --output="$OPD_ONLINE_RUN/logs/%j.out" --error="$OPD_ONLINE_RUN/logs/%j.err" \
 "$OPD_CODE_ROOT/ocrd_pipeline/online_rollout/launch_workers_2nodes.slurm")
printf '%s\n' "$gpu_job" > "$OPD_ONLINE_RUN/gpu_job.id"
nohup bash "$OPD_CODE_ROOT/ocrd_pipeline/online_rollout/launch_coordinator_2nodes.sh" \
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
           nodes=['gnho011','gnho034'],gpus=16,replicas=8,student_gpus=8,teacher_gpus=8,
           concurrency=256,questions=1000,chunk_tokens=64,teacher_tokens=8,
           gap_threshold=2.,rollouts_per_question=1,takeover_count_limit=None,
           resource_ceiling_nodes=2,resource_ceiling_h800=16,
           migration=json.loads((root/'results/migration.json').read_text())['counts'])
(root/'launch.json').write_text(json.dumps(value,indent=2))
print(json.dumps(value))
PY
