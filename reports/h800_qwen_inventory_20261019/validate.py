paths=['/mnt/awomo-data/experiments/ap_feedback_memory_20261002/storage/yukaichengLab/share/model/Qwen3-4B', '/mnt/awomo-data/experiments/robotwin_eval_only_20261008_h800/payload/mnt/auwomo-data/auwomo-model/Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct', '/mnt/awomo-data/npy/wam_bench/tools/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c', '/mnt/awomo-data/npy/wam_bench/legacy/yukaichengLab/niupenghui/wam_bench/sft_assets/wam_models/base/imagewam/Qwen3-4B']
from pathlib import Path
import json,struct
for path in paths:
 p=Path(path);out=[]
 for f in sorted(p.glob('*.safetensors')):
  with f.open('rb') as h:
   n=struct.unpack('<Q',h.read(8))[0]
   assert n<16000000,(f,n)
   header=json.loads(h.read(n))
  tensors={k:v for k,v in header.items() if k!='__metadata__'}
  end=max(v['data_offsets'][1] for v in tensors.values())
  assert 8+n+end==f.stat().st_size,(str(f),'size mismatch')
  out.append({'file':f.name,'tensors':len(tensors),'bytes':f.stat().st_size,'header_and_size_valid':True})
 print(json.dumps({'path':path,'validated_weights':out}),flush=True)
