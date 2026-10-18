import sys,runpy,json
from pathlib import Path
C=Path('/storage/yukaichengLab/wangxin/test_eiloeo/opd_online_code/20260911_chunk64_k8_2nodes/ocrd_pipeline')
P=Path('/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_hard/hard_k32_20260911')
sys.path.insert(0,str(C))
from online_rollout import core
original=core.rollout
policy=json.loads((P/'USER_ABORT_POLICY.json').read_text())
def rollout(row,settings,backend,judge,checkpoint=None,save=lambda _:None):
 if row['dataset_index'] in policy['indices']:
  assert checkpoint is not None
  state=dict(checkpoint,status='USER_ABORTED',finish_reason='user_abandoned_long_trace')
  save(state)
  return state
 return original(row,settings,backend,judge,checkpoint,save)
core.rollout=rollout
runpy.run_path(str(C/'online_rollout/run.py'),run_name='__main__')
