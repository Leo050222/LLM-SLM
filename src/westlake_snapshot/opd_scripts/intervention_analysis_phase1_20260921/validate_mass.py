import json
import numpy as np
import pandas as pd
from safetensors import safe_open
from extract import OUT,ARCHIVE,save

df=pd.read_parquet(OUT/'events.parquet')
selected={r['event_id'] for r in json.loads((OUT/'INDEPENDENT_VALIDATION.json').read_text())['results']}
rows=[]
for shard in json.loads((ARCHIVE/'manifest.json').read_text())['shards']:
    path=ARCHIVE/'student'/shard['name'].replace('.jsonl','.safetensors')
    with safe_open(str(path),framework='pt',device='cpu') as f:
        ids=f.get_tensor('event_ids').tolist()
        for offset,event in enumerate(ids):
            if event not in selected:continue
            r=df[df.event_id==event].iloc[0]
            z=f.get_slice('logits')[offset:offset+1].float().numpy()[0].astype(np.float64)
            p=np.exp(z-z.max());p/=p.sum();y=int(r.teacher_token_id);x=int(r.student_token_id)
            lo=p[z>z[y]].sum();hi=lo+p[z==z[y]].sum()-p[y]
            errlo=abs(lo-r.w_mass_above);errhi=abs(hi-(r.w_mass_above+r.w_tie_mass-r.w_py))
            assert max(errlo,errhi)<2e-6
            assert abs((z[x]-z[y])-(r.w_logpx-r.w_logpy))<2e-5
            rows.append(dict(event_id=event,mass_lo_error=errlo,mass_hi_error=errhi))
assert len(rows)==40
save(OUT/'continuous_support'/'INDEPENDENT_VALIDATION.json',dict(status='PASS',events=len(rows),
    max_mass_error=max(max(r['mass_lo_error'],r['mass_hi_error']) for r in rows),results=rows))
print('MASS_FP64_VALIDATION_PASS',len(rows))
