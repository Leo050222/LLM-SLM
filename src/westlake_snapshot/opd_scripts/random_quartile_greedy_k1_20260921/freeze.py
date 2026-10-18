import json, hashlib, shutil
from pathlib import Path
R=Path('/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_ablation')
S=R/'random_quartile_k1_20260918'
P=R/'random_quartile_greedy_k1_20260921'
P.mkdir(exist_ok=True)
m=json.loads((S/'INPUTS_COMPLETE.json').read_text())
assert hashlib.sha256((S/'prompts.jsonl').read_bytes()).hexdigest()==m['sha256']
assert m['rows']==4571
for name in ['prompts.jsonl','INPUTS_COMPLETE.json','source_manifests.json','base_settings.json','configs.json']:
 data=(S/name).read_bytes()
 if (P/name).exists(): assert (P/name).read_bytes()==data
 else: (P/name).write_bytes(data)
print(json.dumps(m))
