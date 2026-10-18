import json
from pathlib import Path
import sys
import time
root = Path(sys.argv[1])
traces = [json.loads(p.read_text()) for p in (root/'results/traces').glob('*.json')]
judges = [json.loads(p.read_text()) for p in (root/'results/judge_cache').glob('*.json')]
manifest = root/'results/manifest.json'
elapsed = time.time() - manifest.stat().st_mtime if manifest.exists() else 0
report = dict(elapsed_seconds=elapsed, traces=len(traces), completed=sum(r['status']=='COMPLETE' for r in traces),
              tokens=sum(len(r['response_token_ids']) for r in traces),
              takeovers=sum(sum('bridge_token_ids' in e for e in r['events']) for r in traces),
              max_takeovers=max([sum('bridge_token_ids' in e for e in r['events']) for r in traces] or [0]),
              judge_calls=len(judges),
              mean_judge_seconds=sum(j.get('metadata',{}).get('elapsed_seconds',0) for j in judges)/max(1,len(judges)))
for role in ('student','teacher'):
    paths=list((root/'queue'/role/'results').glob('*.json'))
    results=[json.loads(p.read_text()) for p in paths]
    report[role]=dict(requests=len(results),errors=sum('error' in r for r in results),
                      batch_gpu_seconds=sum(r.get('worker_batch_seconds',0)/r.get('worker_batch_size',1) for r in results),
                      pending=len(list((root/'queue'/role/'requests').glob('*.json'))))
print(json.dumps(report,indent=2))
