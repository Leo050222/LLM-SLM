import asyncio,tempfile,time,unittest
from pathlib import Path
from queue_rpc import atomic
from async_service import serve
class Tests(unittest.IsolatedAsyncioTestCase):
 async def test_refill_before_long_finishes_and_no_duplicates(self):
  with tempfile.TemporaryDirectory() as d:
   base=Path(d);started=[];finished=[];active=0;peak=0
   for key,duration in [('a',.5),('b',.03),('c',.03)]:
    atomic(base/'requests'/f'{key}.json',{'duration':duration});atomic(base/'leases'/key/'a.json',{'expires_at':time.time()+10})
   async def execute(req,key,wait):
    nonlocal active,peak
    started.append(key);active+=1;peak=max(peak,active)
    try:await asyncio.sleep(req['duration']);finished.append(key);return {'ok':1}
    finally:active-=1
   task=asyncio.create_task(serve(base,base/'STOP',execute,2))
   await asyncio.sleep(.25)
   self.assertEqual(started,['a','b','c']);self.assertEqual(finished,['b','c']);self.assertEqual(peak,2)
   self.assertTrue((base/'results/c.json').exists());self.assertFalse((base/'results/a.json').exists())
   await asyncio.sleep(.4);(base/'STOP').touch();await task;self.assertEqual(started.count('a'),1)
 async def test_last_lease_expiry_cancels_and_rejoin_reexecutes(self):
  with tempfile.TemporaryDirectory() as d:
   base=Path(d);cancelled=asyncio.Event();calls=[]
   atomic(base/'requests/a.json',{'op':'score'});atomic(base/'leases/a/first.json',{'expires_at':time.time()+.08})
   async def execute(req,key,wait):
    calls.append(key)
    try:await asyncio.sleep(10)
    except asyncio.CancelledError:cancelled.set();raise
   task=asyncio.create_task(serve(base,base/'STOP',execute,1))
   await asyncio.wait_for(cancelled.wait(),2)
   self.assertFalse((base/'results/a.json').exists())
   atomic(base/'leases/a/second.json',{'expires_at':time.time()+10});atomic(base/'requests/a.json',{'op':'score'})
   await asyncio.sleep(.1);self.assertEqual(len(calls),2)
   (base/'STOP').touch();await task
 async def test_one_expired_waiter_does_not_cancel_live_waiter(self):
  with tempfile.TemporaryDirectory() as d:
   base=Path(d);cancelled=[]
   atomic(base/'requests/a.json',{'op':'score'})
   atomic(base/'leases/a/first.json',{'expires_at':time.time()+.03});atomic(base/'leases/a/second.json',{'expires_at':time.time()+3})
   async def execute(req,key,wait):
    try:await asyncio.sleep(.4);return {'ok':1}
    except asyncio.CancelledError:cancelled.append(key);raise
   task=asyncio.create_task(serve(base,base/'STOP',execute,1));await asyncio.sleep(.6)
   self.assertTrue((base/'results/a.json').exists());self.assertEqual(cancelled,[])
   (base/'STOP').touch();await task
 async def test_failure_propagates_and_stops_other_tasks(self):
  with tempfile.TemporaryDirectory() as d:
   base=Path(d);atomic(base/'requests/a.json',{});atomic(base/'leases/a/a.json',{'expires_at':time.time()+5})
   async def fail(*args):raise ValueError('injected')
   with self.assertRaises(ValueError):await serve(base,base/'STOP',fail,1)
   self.assertTrue((base/'results/a.json').exists())
if __name__=='__main__':unittest.main()
