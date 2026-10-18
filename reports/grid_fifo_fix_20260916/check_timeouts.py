import json
from pathlib import Path
R=Path('/storage/yukaichengLab/wangxin/test_eiloeo');out={}
for kind,rel,job in [('code','opd_eval_code/code_grid_x8_20260916',104996),('math','opd_eval_hard/math_reasoning_grid_x8_20260916',104997)]:
 p=R/rel;items=[]
 for l in (p/'slots'/f'job_{job}'/'logs/dispatch.log').read_text().splitlines():
  a=l.split()
  if a[:1]!=['TRACE_DONE'] or 'trace_timeout' not in a:continue
  g,i=a[1],int(a[2]);f=p/'recovery_fifo_v2/checkpoint_backup'/g/'timings'/f'{i:05d}.json'
  items.append({'group':g,'index':i,'old_elapsed':json.loads(f.read_text())['elapsed_seconds'] if f.exists() else None})
 out[kind]=items
print(json.dumps(out))
