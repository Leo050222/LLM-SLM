"""Read-only partitioned continuation of the checkpointed BOS name search."""
import base64,json,pathlib,subprocess,threading,time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from zoneinfo import ZoneInfo
from check_bos_prefix import BceCredentials,BceClientConfiguration,BosClient,HTTPS,NoRetryPolicy,safe_error,safe_scalar
from baidubce.http import bce_http_client
# Reuse one SDK HTTPS connection per worker; certificate verification and
# request signing remain handled by the original SDK connection factory.
_connection_factory=bce_http_client._get_connection
_connections=threading.local()
def _reused_connection(*args,**kwargs):
 cache=getattr(_connections,'cache',None)
 if cache is None:cache={};_connections.cache=cache
 key=(args,tuple(sorted(kwargs.items())))
 if key not in cache:cache[key]=_connection_factory(*args,**kwargs)
 return cache[key]
bce_http_client._get_connection=_reused_connection
ROOT=pathlib.Path(__file__).parent/'wangxin-full-search-20261011'
LOCK=threading.RLock(); STOP=threading.Event(); STATES={}; SEGMENTS={}
def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def save(p,v):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2));t.replace(p)
def client(region):return BosClient(BceClientConfiguration(credentials=CREDS,endpoint='https://'+region+'.bcebos.com',protocol=HTTPS,connection_timeout_in_mills=60000,retry_policy=NoRetryPolicy()))
def call(c,name,**kw):
 for attempt in range(3):
  try:return c.list_objects(name,**kw)
  except Exception as e:
   if attempt==2 or getattr(e,'status_code',None) in (401,403,404):raise
   time.sleep(1)
def add_objects(name,objects):
 s=STATES[name];ms=[]
 for o in objects:
  k=safe_scalar(o.key)
  if 'wangxin' in k.casefold():ms.append({'bucket':name,'key':k,'size_bytes':o.size,'last_modified':safe_scalar(getattr(o,'last_modified',None))})
 with LOCK:
  s['objects_scanned']+=len(objects);s['matches']+=len(ms);s['updated_at']=now()
  if ms:
   with (ROOT/(name+'-matches.jsonl')).open('a') as f:
    for m in ms:f.write(json.dumps(m,ensure_ascii=False)+'\n')
  save(ROOT/(name+'.json'),s)
 if ms:print(json.dumps({'new_matches':len(ms),'bucket':name,'first':ms[0]['key']},ensure_ascii=False),flush=True)
def discover(name):
 s=STATES[name];sf=ROOT/(name+'-segments.json')
 if sf.exists():
  seg=json.loads(sf.read_text())
  for i,g in enumerate(seg):
   latest=ROOT/(name+'-segment-'+str(g['id'])+'.json')
   if latest.exists():seg[i]=json.loads(latest.read_text())
   if seg[i]['status']=='error':seg[i]['status']='pending'
  SEGMENTS[name]=seg;return
 c=client(s['region']);cut=s.get('next_marker') or ''; direct=[]
 def children(prefix):
  ps=[];marker=None
  while True:
   r=call(c,name,prefix=prefix,delimiter='/',max_keys=1000,marker=marker)
   direct.extend(o for o in (getattr(r,'contents',None) or []) if safe_scalar(o.key)>cut)
   for x in (getattr(r,'common_prefixes',None) or []):
    p=safe_scalar(x.prefix)
    if p>cut or cut.startswith(p):ps.append(p)
   s['discovery_pages']=s.get('discovery_pages',0)+1
   if not r.is_truncated:break
   nm=safe_scalar(getattr(r,'next_marker',None))
   if not nm or nm==marker:raise RuntimeError('Directory pagination did not advance')
   marker=nm
  return ps
 prefixes=children('')
 # Split every remaining root once, and follow the current marker's directory
 # until the hot path has enough independent children.
 result=[]
 def expand(prefix,depth=0,force=False):
  if depth>24:result.append(prefix);return
  ps=children(prefix)
  if not ps:return
  if cut.startswith(prefix):
   for p in ps:
    if cut.startswith(p):expand(p,depth+1,True)
    else:result.append(p)
  elif len(ps)==1 and depth<4:expand(ps[0],depth+1)
  else:result.extend(ps)
 for p in prefixes:expand(p)
 # Empty buckets with direct objects are handled by discovery itself.
 add_objects(name,direct)
 seg=[{'id':i,'prefix':p,'next_marker':cut if cut.startswith(p) else None,'status':'pending','objects_scanned':0,'pages_scanned':0} for i,p in enumerate(result)]
 SEGMENTS[name]=seg;s['partitioned']=True;s['segments_total']=len(seg);s['next_marker']=None
 with LOCK:save(sf,seg);save(ROOT/(name+'.json'),s)
 print(json.dumps({'event':'partitioned','bucket':name,'segments':len(seg),'direct_objects':len(direct)}),flush=True)
def scan(task):
 name,g=task;s=STATES[name];c=client(s['region'])
 if g['status']=='complete':return
 g['status']='running'
 try:
  if g.get('include_start'):
   r=call(c,name,prefix=g['next_marker'],max_keys=1)
   exact=[o for o in (getattr(r,'contents',None) or []) if safe_scalar(o.key)==g['next_marker']]
   add_objects(name,exact);g['include_start']=False
   with LOCK:save(ROOT/(name+'-segment-'+str(g['id'])+'.json'),g)
  while True:
   marker=g['next_marker'];r=call(c,name,prefix=g['prefix'],marker=marker,max_keys=1000)
   if getattr(r,'common_prefixes',None):raise RuntimeError('Unexpected directory-only response')
   os=getattr(r,'contents',None) or []
   upper=g.get('upper_bound');reached_end=bool(upper and any(safe_scalar(o.key)>=upper for o in os))
   if upper:os=[o for o in os if safe_scalar(o.key)<upper]
   add_objects(name,os)
   truncated=bool(r.is_truncated) and not reached_end;nm=safe_scalar(getattr(r,'next_marker',None))
   if truncated and not nm and os:nm=safe_scalar(os[-1].key)
   with LOCK:
    g['objects_scanned']+=len(os);g['pages_scanned']+=1;s['pages_scanned']+=1;g['next_marker']=nm;g['status']='running' if truncated else 'complete'
    if truncated and (not nm or nm==marker):raise RuntimeError('Pagination marker did not advance')
    save(ROOT/(name+'-segment-'+str(g['id'])+'.json'),g)
   if not truncated:break
 except Exception as e:g['status']='error';g['error']=safe_error(e);print(json.dumps({'bucket':name,'prefix':g['prefix'],'error':g['error']}),flush=True)
def snapshot():
 with LOCK:
  for n,gs in SEGMENTS.items():
   s=STATES[n];s['segments_complete']=sum(g['status']=='complete' for g in gs)
   if all(g['status']=='complete' for g in gs):s['status']='complete'
   elif any(g['status']=='error' for g in gs):s['status']='error'
   else:s['status']='running'
   save(ROOT/(n+'-segments.json'),gs);save(ROOT/(n+'.json'),s)
  rows=[dict(STATES[k]) for k in sorted(STATES)]
  v={'updated_at':now(),'read_only':True,'search':'case-insensitive substring wangxin in all object keys','file_contents_scanned':False,'completed_buckets':sum(s['status']=='complete' for s in rows),'total_buckets':len(rows),'objects_scanned':sum(s['objects_scanned'] for s in rows),'matches':sum(s['matches'] for s in rows),'buckets':rows}
  save(ROOT/'summary.json',v)
 print(json.dumps({k:v[k] for k in ['updated_at','completed_buckets','total_buckets','objects_scanned','matches']}),flush=True)
def progress():
 while not STOP.wait(25):snapshot()
def main():
 global CREDS
 r=subprocess.run(['kubectl','--kubeconfig','/Users/leolee/Downloads/auwomo-h800-team','--context','cluster1-team-auwomo','--request-timeout=15s','-n','dcloud-auwomo','get','secret','jfs-private-prod-awomo-self-hosted-secret','-o','json'],capture_output=True,text=True,check=True)
 d=json.loads(r.stdout)['data'];CREDS=BceCredentials(base64.b64decode(d['access-key']),base64.b64decode(d['secret-key']))
 for p in ROOT.glob('*.json'):
  if p.stem=='summary' or '-segment' in p.stem:continue
  s=json.loads(p.read_text())
  if not isinstance(s,dict) or p.stem!=s.get('bucket'):continue
  STATES[s['bucket']]=s
 pending=[n for n,s in STATES.items() if s['status']!='complete']
 with ThreadPoolExecutor(max_workers=4) as ex:list(ex.map(discover,pending))
 ts=[(n,g) for n,gs in SEGMENTS.items() for g in gs if g['status']!='complete'];snapshot();t=threading.Thread(target=progress,daemon=True);t.start()
 with ThreadPoolExecutor(max_workers=24) as ex:list(ex.map(scan,ts))
 STOP.set();t.join();snapshot();print('SEARCH_FINISHED',flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:print(json.dumps({'fatal_error':safe_error(e)}),flush=True);raise SystemExit(1)
