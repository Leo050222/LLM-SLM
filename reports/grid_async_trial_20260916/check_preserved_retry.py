import json,hashlib
from pathlib import Path
R=Path('/storage/yukaichengLab/wangxin/test_eiloeo');out={}
for kind,rel in [('math','opd_eval_hard/math_reasoning_grid_x8_20260916')]:
 p=R/rel;rec=p/'recovery_async_retry_v3';hashes=json.loads((rec/'preserved_hashes.json').read_text());n=0;timeouts=0
 for name,h in hashes.items():
  if '/completed/' not in name:continue
  raw=(p/name).read_bytes();assert hashlib.sha256(raw).hexdigest()==h,name
  n+=1;timeouts+=json.loads(raw).get('finish_reason')=='trace_timeout'
 out[kind]={'preserved_completed':n,'preserved_timeouts':timeouts,'all_hashes_unchanged':True,'archived_files':len(hashes)}
print(json.dumps(out))
