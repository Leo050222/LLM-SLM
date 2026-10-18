import json
from pathlib import Path
p=Path('/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_ablation/weak_reprompt_t2_k1_20260920')
s=p/'slots/job_107636'
print('ready',len(list((s/'queue').glob('replica_*/*/ready.json'))))
for name in ['smoke','runtime_smoke','dispatch']:
 f=s/'logs'/f'{name}.log'
 print(name, f.read_text()[-2400:] if f.exists() else 'not_started')
for f in sorted((s/'logs').glob('gpu*.log')):
 lines=f.read_text().splitlines();print(f.name, '\n'.join(lines[-2:]))
print('completed',len(list((p/'r1_weak_reprompt_t2_k1/completed').glob('*.json'))))
