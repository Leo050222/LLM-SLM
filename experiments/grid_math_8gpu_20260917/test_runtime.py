import unittest,tempfile,time,threading,json,os
from pathlib import Path
from queue_rpc import RPC,atomic,claim_batch,publish
from deadline import Budget,TraceDeadline
from migration import compatible
class Tests(unittest.TestCase):
 def test_fifo_op_fairness(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)
   for name,age,op in [('ff',10,'score'),('00',1,'generate'),('01',.5,'score')]:
    f=p/'requests'/f'{name}.json';atomic(f,{'op':op});os.utime(f,(time.time()-age,)*2);atomic(p/'leases'/name/'a.json',{'expires_at':time.time()+20})
   b=claim_batch(p,8);self.assertEqual([x[0].stem for x in b],['ff','01'])
   for path,_,_ in b:publish(p,path,{'ok':1})
   self.assertEqual(claim_batch(p,8)[0][0].stem,'00')
 def test_shared_waiter(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);results={}
   def call(name,limit):
    try:results[name]=RPC(p,Budget(p/f'{name}.json',limit)).call('teacher',{'op':'score'})
    except TraceDeadline:results[name]='timeout'
   a=threading.Thread(target=call,args=('a',.3));a.start()
   end=time.monotonic()+2
   while not list((p/'teacher/requests').glob('*.json')) and time.monotonic()<end:time.sleep(.005)
   request=next((p/'teacher/requests').glob('*.json'));stamp=request.stat().st_mtime_ns
   b=threading.Thread(target=call,args=('b',3));b.start();a.join(2)
   self.assertFalse(a.is_alive());self.assertTrue(request.exists());self.assertEqual(request.stat().st_mtime_ns,stamp)
   publish(p/'teacher',request,{'ok':1});b.join(2);self.assertFalse(b.is_alive());self.assertEqual(results['a'],'timeout');self.assertEqual(results['b']['ok'],1)
 def test_orphan_cleanup(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)
   with self.assertRaises(TraceDeadline):RPC(p,Budget(p/'timer.json',.03)).call('student',{'op':'generate'})
   self.assertEqual(list((p/'student/requests').glob('*')),[])
 def test_expired_not_executed(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);atomic(p/'requests/ff.json',{'op':'score'});atomic(p/'leases/ff/a.json',{'expires_at':0})
   self.assertEqual(claim_batch(p,8),[])
 def test_budget_resume(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'timer';atomic(p,{'elapsed_seconds':901})
   with self.assertRaises(TraceDeadline):Budget(p).check()
 def test_migration_science(self):
  a={'settings':{'seed':1},'models':{r:{'protocol':1,'worker_sha256':'63f31492dbeea547b0917c4ee8525b8a9aa22caecc2fae9c66e5061ff99891b4','model':r} for r in ['student','teacher']}}
  b=json.loads(json.dumps(a))
  for v in b['models'].values():v.update(protocol=2,worker_sha256='new')
  compatible(a,b);b['settings']['seed']=2
  with self.assertRaises(ValueError):compatible(a,b)
if __name__=='__main__':unittest.main()
