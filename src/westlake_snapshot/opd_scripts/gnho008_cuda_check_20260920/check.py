import concurrent.futures
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

out=Path('/storage/yukaichengLab/wangxin/test_eiloeo/opd_profile/gnho008_cuda_check_20260920')/os.environ['SLURM_JOB_ID']
out.mkdir(parents=True,exist_ok=True)
smi=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,memory.total','--format=csv'],capture_output=True,text=True,timeout=30)
print(smi.stdout,smi.stderr,flush=True)
devices=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
assert len(devices)==8,devices
allocated_devices=devices
inventory=list(csv.DictReader(smi.stdout.splitlines(),skipinitialspace=True))
assert smi.returncode==0 and len(inventory)==8
devices=[r['uuid'] for r in inventory]
code='''
import json,torch
torch.manual_seed(20260920)
assert torch.cuda.device_count()==1
p=torch.cuda.get_device_properties(0)
free,total=torch.cuda.mem_get_info()
assert free>4*1024**3
x=torch.randn(1024,1024,device='cuda')
f=x@x
assert torch.isfinite(f).all()
b=x.to(torch.bfloat16)
y=b@b
assert y.dtype==torch.bfloat16 and torch.isfinite(y).all()
q=torch.randn(1,8,128,128,device='cuda',dtype=torch.bfloat16)
a=torch.nn.functional.scaled_dot_product_attention(q,q,q,is_causal=True)
assert torch.isfinite(a).all()
torch.cuda.synchronize()
print(json.dumps(dict(ok=True,name=p.name,uuid=str(p.uuid),free_bytes=free,total_bytes=total,fp32_checksum=f.double().sum().item(),bf16_checksum=y.double().sum().item(),sdpa_checksum=a.double().sum().item(),torch=torch.__version__)))
'''
def test(d):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=d,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    try:
        p=subprocess.run([sys.executable,'-c',code],env=env,capture_output=True,text=True,timeout=90)
        r=dict(device=d,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
    except Exception as e:r=dict(device=d,returncode=-1,error=str(e))
    print(json.dumps(r),flush=True)
    return r
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    results=list(pool.map(test,devices))
report=dict(job_id=os.environ['SLURM_JOB_ID'],node=os.environ['SLURMD_NODENAME'],devices=devices,allocated_devices=allocated_devices,inventory=inventory,
            nvidia_smi=dict(returncode=smi.returncode,stdout=smi.stdout,stderr=smi.stderr),
            results=results,all_passed=all(r['returncode']==0 for r in results),time=time.time())
(out/'result.json').write_text(json.dumps(report,indent=2))
print('ALL_PASSED',report['all_passed'],flush=True)
