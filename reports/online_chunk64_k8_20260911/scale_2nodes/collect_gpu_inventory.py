import csv,json,socket,subprocess,sys
from pathlib import Path
text=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
rows=[dict(pid=int(row[0].strip()),uuid=row[1].strip(),memory_mib=int(row[2].strip())) for row in csv.reader(text.splitlines())]
node=socket.gethostname().split('.')[0]
root=Path(sys.argv[1])/'gpu_inventory'; root.mkdir(exist_ok=True)
(root/(node+'.json')).write_text(json.dumps(rows,indent=2))
print(node,len({row['uuid'] for row in rows}))
