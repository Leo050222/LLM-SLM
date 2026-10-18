"""Independent float64 checks against selected original BF16 archive rows."""
import json
import numpy as np
import pandas as pd
from safetensors import safe_open
from extract import OUT, ARCHIVE, save

df=pd.read_parquet(OUT/'events.parquet')
tr=pd.read_parquet(OUT/'traces.parquet')
selected=set()
for _,g in df.groupby('dataset'):
    selected.update([int(g.event_id.iloc[0]),int(g.event_id.iloc[-1]),
                     int(g.loc[g.w_py.idxmin(),'event_id']),int(g.loc[g.w_py.idxmax(),'event_id']),
                     int(g.loc[g.w_tie_count.idxmax(),'event_id'])])
records=[]
manifest=json.loads((ARCHIVE/'manifest.json').read_text())
for shard in manifest['shards']:
    name=shard['name'].replace('.jsonl','.safetensors')
    with safe_open(str(ARCHIVE/'student'/name),framework='pt',device='cpu') as f:
        ids=f.get_tensor('event_ids').tolist()
    take=[(i,e) for i,e in enumerate(ids) if e in selected]
    if not take:continue
    for role,tag in [('student','w'),('teacher','s')]:
        with safe_open(str(ARCHIVE/role/name),framework='pt',device='cpu') as f:
            for offset,event in take:
                row=df[df.event_id==event].iloc[0]
                z=f.get_slice('logits')[offset:offset+1].float().numpy()[0].astype(np.float64)
                x,y=int(row.student_token_id),int(row.teacher_token_id)
                lp=z-z.max()-np.log(np.exp(z-z.max()).sum())
                p=np.exp(lp);h=-(p*lp).sum()
                masked=np.delete(z,x);mlp=masked-masked.max()-np.log(np.exp(masked-masked.max()).sum())
                mh=-(np.exp(mlp)*mlp).sum();qy=np.exp(mlp[y-(x<y)])
                checks={'logpy':lp[y],'py':p[y],'entropy':h,'masked_entropy':mh,'qy':qy,
                        'rank_min':1+int((z>z[y]).sum()),'rank_max':int((z>=z[y]).sum())}
                for key,value in checks.items():
                    got=float(row[tag+'_'+key]);err=abs(got-value)
                    assert np.isclose(got,value,rtol=2e-5,atol=2e-5),(event,tag,key,got,value)
                    records.append(dict(event_id=event,role=role,metric=key,abs_error=err))
assert len(records)==len(selected)*2*7
assert len(tr)==36568 and tr.interventions.sum()==len(df)==399601
assert (tr.interventions==0).sum()==816
assert tr.finish_reason.eq('trace_timeout').sum()==227
assert not df.correct.isna().any() and not tr.correct.isna().any()
save(OUT/'INDEPENDENT_VALIDATION.json',dict(status='PASS',events=len(selected),checks=len(records),
    method='Original BF16 rows, independent NumPy float64 normalization, masked vector deletion, exact tie counts',
    max_abs_error=max(r['abs_error'] for r in records),results=records))
print('INDEPENDENT_VALIDATION_PASS',len(selected),'events',len(records),'checks',flush=True)
