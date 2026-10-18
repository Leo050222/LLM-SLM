"""Select only allocated, free, healthy GPUs; bind UUIDs, never CUDA ordinals."""
import os, subprocess, json
import xml.etree.ElementTree as ET
from pathlib import Path
raw_ids=os.environ.get('SLURM_STEP_GPUS') or os.environ.get('SLURM_JOB_GPUS')
assert raw_ids, 'Missing Slurm global GPU allocation IDs'
allocated=set()
for part in raw_ids.split(','):
 if '-' in part:
  a,b=map(int,part.split('-'));allocated.update(range(a,b+1))
 else:allocated.add(int(part))
xml=ET.fromstring(subprocess.check_output(['nvidia-smi','-q','-x'],text=True))
minors={g.findtext('uuid'):int(g.findtext('minor_number')) for g in xml.findall('gpu')}
raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free','--format=csv,noheader,nounits'],text=True)
selected=[];inventory=[]
for line in raw.splitlines():
 idx,uuid,used,free=[v.strip() for v in line.split(',')]
 minor=minors[uuid]
 item=dict(physical_index=int(idx),minor_number=int(minor),uuid=uuid,used_mib=int(used),free_mib=int(free));inventory.append(item)
 if int(minor) in allocated and int(idx)!=2 and int(used)<500 and int(free)>=74000:
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=idx,CUDA_DEVICE_ORDER="PCI_BUS_ID",EXPECTED_GPU_UUID=uuid)
  result=subprocess.run([os.environ['OPD_PYTHON'],'-c','import torch,os; actual=str(torch.cuda.get_device_properties(0).uuid); assert actual.removeprefix("GPU-").lower()==os.environ["EXPECTED_GPU_UUID"].removeprefix("GPU-").lower(), (actual,os.environ["EXPECTED_GPU_UUID"]); torch.manual_seed(20260920); x=torch.randn(64,64,device="cuda",dtype=torch.bfloat16); y=x@x; torch.cuda.synchronize(); assert torch.isfinite(y).all(); print("GPU_SELECT_WITNESS", torch.cuda.get_device_name())'],env=env,text=True,capture_output=True,timeout=60)
  item['probe_returncode']=result.returncode;item['probe_stdout']=result.stdout;item['probe_stderr']=result.stderr
  if result.returncode==0:selected.append(item)
record=dict(slurm_gpu_ids=raw_ids,allocated_minor_numbers=sorted(allocated),inventory=inventory,selected=selected)
p=Path(os.environ['OPD_ONLINE_RUN']);(p/'GPU_SELECTION.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record),flush=True)
assert selected, 'No allocated free healthy GPU passed CUDA witness'
(p/'healthy_gpu_uuids.txt').write_text(''.join(x['uuid']+'\n' for x in selected))

(p/'healthy_gpu_indices.txt').write_text(''.join(str(x['physical_index'])+'\n' for x in selected))
