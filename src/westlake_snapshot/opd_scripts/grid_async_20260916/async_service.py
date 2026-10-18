"""Bounded per-request completion with FIFO admission and shared-waiter leases."""
import asyncio,json,time
from queue_rpc import locked,live,publish

def claim_ready(base,capacity,inflight):
 with locked(base):
  ready=[]
  for path in (base/'requests').glob('*.json'):
   if path.stem in inflight:continue
   if (base/'results'/path.name).exists() or not live(base,path.stem,time.time()):
    path.unlink(missing_ok=True);continue
   ready.append((path.stat().st_mtime_ns,path))
  ready.sort(key=lambda x:(x[0],x[1].name))
  return [(p,json.loads(p.read_text()),max(0,time.time()-stamp/1e9)) for stamp,p in ready[:capacity]]

def expired(base,keys):
 with locked(base):
  return [k for k in keys if not live(base,k,time.time())]

async def serve(base,stop_path,execute,capacity=8):
 tasks={};last_leases=0
 async def one(path,payload,wait):
  try:
   value=await execute(payload,path.stem,wait)
   await asyncio.to_thread(publish,base,path,value)
  except asyncio.CancelledError:
   # No error result: a new waiter may join after the cancellation decision.
   raise
  except Exception as e:
   await asyncio.to_thread(publish,base,path,{'error':f'{type(e).__name__}: {e}'})
   raise
 try:
  while not await asyncio.to_thread(stop_path.exists):
   for key,task in list(tasks.items()):
    if task.done():
     del tasks[key]
     if not task.cancelled():task.result()
   if tasks and time.monotonic()-last_leases>=.2:
    dead=await asyncio.to_thread(expired,base,list(tasks))
    for key in dead:
     if not tasks[key].done() and not tasks[key].cancelling():tasks[key].cancel()
    last_leases=time.monotonic()
   if len(tasks)<capacity:
    batch=await asyncio.to_thread(claim_ready,base,capacity-len(tasks),set(tasks))
    for path,payload,wait in batch:
     assert path.stem not in tasks
     tasks[path.stem]=asyncio.create_task(one(path,payload,wait))
   await asyncio.sleep(.02)
 finally:
  for task in tasks.values():task.cancel()
  results=await asyncio.gather(*tasks.values(),return_exceptions=True)
