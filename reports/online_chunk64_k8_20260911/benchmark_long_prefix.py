import json
from pathlib import Path
import sys
import time
sys.path.insert(0, sys.argv[1] + '/ocrd_pipeline')
from online_rollout.core import FileRPC, atomic_json, digest
root=Path(sys.argv[2])
source=Path('/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_deepmath/ge6_test1000_native_nonthinking_seed20260811/qwen3_1_7b/merged/rollouts.jsonl')
# A fixed pre-existing baseline prefix is used only for throughput diagnostics.
with source.open() as handle:
    row=max((json.loads(line) for line in handle),key=lambda r:len(r['response_token_ids']))
full=row['prompt_token_ids']+row['response_token_ids']
# Repeat only for synthetic long-context throughput; never an evaluation trace.
full=(full*((32832+len(full)-1)//len(full)))[:32832]
rpc=FileRPC(root/'queue')
results=[]
for length in (1024,8192,16384,32768):
    for role in ('student','teacher'):
        model=json.loads((root/'queue'/role/'model.json').read_text())
        started=time.monotonic()
        result=rpc.call(role,dict(model_identity=digest(model),op='score',prefix=full[:length],tokens=full[length:length+64]))
        assert len(result['chosen'])==64
        results.append(dict(prefix_tokens=length,role=role,wall_seconds=time.monotonic()-started,
                            worker_batch_seconds=result['worker_batch_seconds'],batch_size=result['worker_batch_size']))
        print(json.dumps(results[-1]),flush=True)
atomic_json(root/'long_prefix_benchmark.json',dict(evaluation_result=False,results=results))
