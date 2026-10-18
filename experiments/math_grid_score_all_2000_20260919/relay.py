"""Authenticated HTTP CONNECT relay restricted to api.openlux.ai:443; TLS end-to-end."""
import asyncio,base64,hmac,json,os,resource,time
from pathlib import Path
P=Path('/storage/yukaichengLab/wangxin/test_eiloeo/opd_rescore/math_grid_score_all_2000_20260919/judge')
expected='Basic '+base64.b64encode(('opd:'+ (P/'.relay_token').read_text().strip()).encode()).decode()
resource.setrlimit(resource.RLIMIT_NOFILE,(min(32768,resource.getrlimit(resource.RLIMIT_NOFILE)[1]),resource.getrlimit(resource.RLIMIT_NOFILE)[1]))
async def pipe(r,w):
 while data:=await r.read(65536):w.write(data);await w.drain()
async def handle(r,w):
 up=None
 try:
  header=(await asyncio.wait_for(r.readuntil(b'\r\n\r\n'),10)).decode('ascii');lines=header.split('\r\n');headers={k.lower():v.strip() for k,v in (line.split(':',1) for line in lines[1:] if ':' in line)}
  if lines[0]!='CONNECT api.openlux.ai:443 HTTP/1.1' or not hmac.compare_digest(headers.get('proxy-authorization',''),expected):
   w.write(b'HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n');await w.drain();return
  remote,up=await asyncio.wait_for(asyncio.open_connection('api.openlux.ai',443),30)
  w.write(b'HTTP/1.1 200 Connection Established\r\n\r\n');await w.drain()
  tasks=[asyncio.create_task(pipe(r,up)),asyncio.create_task(pipe(remote,w))];_,pending=await asyncio.wait(tasks,return_when=asyncio.FIRST_COMPLETED)
  for t in pending:t.cancel()
  await asyncio.gather(*tasks,return_exceptions=True)
 except Exception:pass
 finally:
  if up:up.close()
  w.close()
async def main():
 server=await asyncio.start_server(handle,'192.168.180.240',0,backlog=4096);port=server.sockets[0].getsockname()[1]
 (P/'relay.json').write_text(json.dumps(dict(host='192.168.180.240',port=port,pid=os.getpid(),destination='api.openlux.ai:443')));print('RELAY_READY',port,flush=True)
 started=time.monotonic()
 async with server:
  while not (P/'relay.stop').exists() and time.monotonic()-started<7*86400:await asyncio.sleep(5)
asyncio.run(main())
