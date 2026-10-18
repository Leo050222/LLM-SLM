"""Split only unsearched suffixes of large directory checkpoints."""
import pathlib,json,subprocess,base64
import search_remaining_bos_wangxin as m

def main():
 r=subprocess.run(['kubectl','--kubeconfig','/Users/leolee/Downloads/auwomo-h800-team','--context','cluster1-team-auwomo','--request-timeout=15s','-n','dcloud-auwomo','get','secret','jfs-private-prod-awomo-self-hosted-secret','-o','json'],capture_output=True,text=True,check=True)
 d=json.loads(r.stdout)['data'];m.CREDS=m.BceCredentials(base64.b64decode(d['access-key']),base64.b64decode(d['secret-key']))
 for p in m.ROOT.glob('*.json'):
  s=json.loads(p.read_text())
  if isinstance(s,dict) and 'bucket' in s and 'objects_scanned' in s and 'prefix' not in s:m.STATES[s['bucket']]=s
 for name,s in list(m.STATES.items()):
  if s['status']=='complete':continue
  sf=m.ROOT/(name+'-segments.json');gs=json.loads(sf.read_text())
  for i,g in enumerate(gs):
   p=m.ROOT/(name+'-segment-'+str(g['id'])+'.json')
   if p.exists():gs[i]=json.loads(p.read_text())
  m.SEGMENTS[name]=gs;c=m.client(s['region']);next_id=max(g['id'] for g in gs)+1
  for parent in list(gs):
   if parent['status']=='complete' or parent.get('upper_bound') or parent.get('subdivided'):continue
   cut=parent.get('next_marker') or '';direct=[];pieces=[]
   def ranges(prefix):
    bounds=[prefix+x for x in ['0','A','G','M','S','Z','a','g','m','s','z','\u0080']]
    bounds=[b for b in bounds if b>cut]
    start=cut or None
    for upper in bounds+[None]:
     pieces.append({'prefix':prefix,'next_marker':start,'upper_bound':upper,'include_start':bool(start and start!=cut)})
     start=upper
   def walk(prefix,depth=0):
    if depth>28:ranges(prefix);return
    marker=None;ps=[];local=[]
    while True:
     r=m.call(c,name,prefix=prefix,delimiter='/',max_keys=1000,marker=marker)
     objs=getattr(r,'contents',None) or [];dirs=getattr(r,'common_prefixes',None) or []
     # A large flat leaf is split by ordered key ranges, including exact boundaries.
     if marker is None and r.is_truncated and objs:
      ranges(prefix);return
     local.extend(o for o in objs if m.safe_scalar(o.key)>cut)
     ps.extend(m.safe_scalar(x.prefix) for x in dirs if m.safe_scalar(x.prefix)>cut or cut.startswith(m.safe_scalar(x.prefix)))
     if not r.is_truncated:break
     nm=m.safe_scalar(r.next_marker)
     if not nm or nm==marker:raise RuntimeError('Directory marker did not advance')
     marker=nm
    direct.extend(local)
    for p in ps:
     if cut.startswith(p):walk(p,depth+1)
     else:pieces.append({'prefix':p,'next_marker':None})
   walk(parent['prefix'])
   m.add_objects(name,direct)
   new=[]
   for x in pieces:
    x.update({'id':next_id,'status':'pending','objects_scanned':0,'pages_scanned':0});next_id+=1;new.append(x)
   parent['status']='complete';parent['subdivided']=True;gs.extend(new);s['segments_total']=len(gs)
   m.save(m.ROOT/(name+'-segment-'+str(parent['id'])+'.json'),parent);m.save(sf,gs)
   m.save(m.ROOT/(name+'.json'),s)
   print(json.dumps({'bucket':name,'old_prefix':parent['prefix'],'new_segments':len(new),'direct_objects':len(direct)}),flush=True)
 m.snapshot()
if __name__=='__main__':
 try:main()
 except Exception as e:print(json.dumps({'fatal_error':m.safe_error(e)}),flush=True);raise SystemExit(1)
