"""Authenticated internal TCP relay, restricted to api.openlux.ai:443.

TLS remains end-to-end between the Slurm worker and OpenLux.
The relay never reads the API credential or decrypted API traffic.
"""
import asyncio,hmac,json,os,time
from pathlib import Path
OUT=Path('/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_ablation/shift5_greedy_k1_t2_20260921/judge')
TOKEN=(OUT/'.relay_token').read_bytes()
active=0
last=time.monotonic()

async def pipe(reader,writer):
    while True:
        chunk=await reader.read(65536)
        if not chunk:return
        writer.write(chunk);await writer.drain()

async def handle(reader,writer):
    global active,last
    active+=1;last=time.monotonic();up=None
    try:
        auth=await asyncio.wait_for(reader.readexactly(len(TOKEN)),10)
        if not hmac.compare_digest(auth,TOKEN):return
        remote,up=await asyncio.wait_for(asyncio.open_connection('api.openlux.ai',443),20)
        tasks=[asyncio.create_task(pipe(reader,up)),asyncio.create_task(pipe(remote,writer))]
        _,pending=await asyncio.wait(tasks,return_when=asyncio.FIRST_COMPLETED)
        for task in pending:task.cancel()
        await asyncio.gather(*tasks,return_exceptions=True)
    except Exception:pass
    finally:
        if up:up.close()
        writer.close();active-=1;last=time.monotonic()

async def main():
    server=await asyncio.start_server(handle,'192.168.180.240',0,backlog=256)
    port=server.sockets[0].getsockname()[1]
    (OUT/'relay.json').write_text(json.dumps(dict(host='192.168.180.240',port=port,pid=os.getpid(),destination='api.openlux.ai:443')))
    print('RELAY_READY',port,flush=True)
    started=time.monotonic()
    async with server:
        while not (OUT/'relay.stop').exists() and time.monotonic()-started<7*86400:
            await asyncio.sleep(5)
            if False:break
    print('RELAY_STOPPED',flush=True)

asyncio.run(main())
