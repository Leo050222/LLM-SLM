import os,json,time,collections
from pathlib import Path
root=Path('/mnt/awomo-data');q=collections.deque([(root,0)]);start=time.monotonic();seen=0;hits=[];errors=[]
skip={'.git','.minio.sys','.sys','node_modules','site-packages','__pycache__','episodes','videos','images','wandb','tensorboard','logs','blobs'}
while q and time.monotonic()-start<100 and seen<80000:
 p,depth=q.popleft();seen+=1
 try:
  entries=list(os.scandir(p))
 except OSError as e:
  errors.append([str(p),str(e)]);continue
 names={e.name for e in entries}
 if 'qwen' in p.name.lower() or 'config.json' in names:
  config={}
  if 'config.json' in names:
   try:
    raw=json.loads((p/'config.json').read_text());config={k:raw[k] for k in ['model_type','architectures','_name_or_path','hidden_size','num_hidden_layers','vocab_size','torch_dtype'] if k in raw}
   except Exception: pass
  if 'qwen' in str(p).lower() or 'qwen' in json.dumps(config).lower():
   weights=[e for e in entries if e.name.endswith(('.safetensors','.bin','.gguf')) and e.is_file()]
   indexfiles=[e.name for e in entries if e.name.endswith('.index.json')]
   indexes=[]
   for n in indexfiles:
    try:
     idx=json.loads((p/n).read_text()); shards=sorted(set(idx.get('weight_map',{}).values()));indexes.append({'file':n,'shards':len(shards),'missing':[s for s in shards if not (p/s).is_file()],'total_size':idx.get('metadata',{}).get('total_size')})
    except Exception as e:indexes.append({'file':n,'error':str(e)})
   hit={'path':str(p),'config':config,'weight_files':len(weights),'weight_bytes':sum(e.stat().st_size for e in weights),'indexes':indexes,'files':[e.name for e in entries][:50]};hits.append(hit);print(json.dumps({'model_candidate':hit}),flush=True)
 if depth>=9:continue
 for e in entries:
  if e.name in skip:continue
  try:
   if e.is_dir(follow_symlinks=False):q.append((Path(e.path),depth+1))
  except OSError:pass
print(json.dumps({'scan_summary':{'directories_scanned':seen,'seconds':time.monotonic()-start,'remaining':len(q),'depth_limit':9,'skipped_names':sorted(skip),'errors':errors[:30],'hits':len(hits)}}),flush=True)
