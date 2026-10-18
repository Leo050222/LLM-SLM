import json,hashlib,shutil
from pathlib import Path
from ablation_core import atomic_json
R=Path('/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_prefix')
source=R/'prefix_takeover_caps_20260917';target=R/'prefix_takeover_caps5_10_20260917'
def main():
 target.mkdir(parents=True,exist_ok=True)
 meta=json.loads((source/'INPUTS_COMPLETE.json').read_text());raw=(source/'prompts.jsonl').read_bytes()
 assert hashlib.sha256(raw).hexdigest()==meta['sha256'] and meta['rows']==800
 meta.update(caps=list(range(5,11)),expected_outputs=38400,paired_source=str(source))
 if (target/'INPUTS_COMPLETE.json').exists():
  assert json.loads((target/'INPUTS_COMPLETE.json').read_text())==meta
  assert (target/'prompts.jsonl').read_bytes()==raw
 else:
  (target/'prompts.jsonl').write_bytes(raw)
  shutil.copyfile(source/'sampling_audit.json',target/'sampling_audit.json')
  atomic_json(target/'INPUTS_COMPLETE.json',meta)
 print('IDENTICAL_800_PREFIXES_AND_SEEDS',meta['sha256'],'CAPS',meta['caps'],flush=True)
if __name__=='__main__':main()
