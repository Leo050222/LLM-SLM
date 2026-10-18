#!/usr/bin/env python3
"""Own one Slurm node, witness its GPUs, and supervise isolated vLLM replicas."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

CODE = Path(__file__).resolve().parent
RUN = Path('/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_vlm/qwen3vl_instruct_greedy_20261016')
MODELS = Path('/storage/yukaichengLab/share/model')
MODEL_NAMES = ['Qwen3-VL-2B-Instruct', 'Qwen3-VL-4B-Instruct', 'Qwen3-VL-8B-Instruct']
TASK_HASH = '11f7d30530585a4f8476798c160ee5969efc380c3e31afdf2943974389049bbf'


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(tmp, path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def environment(device, suffix):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=device, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               HF_HUB_DISABLE_TELEMETRY='1', TOKENIZERS_PARALLELISM='false',
               OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8',
               VLLM_WORKER_MULTIPROC_METHOD='spawn', PYTHONDONTWRITEBYTECODE='1')
    cache = Path('/tmp') / ('qwen3vl_' + os.environ['SLURM_JOB_ID'] + '_' + suffix)
    cache.mkdir(parents=True, exist_ok=True)
    env.update(TMPDIR=str(cache), TRITON_CACHE_DIR=str(cache / 'triton'),
               VLLM_CACHE_ROOT=str(cache / 'vllm'), TORCHINDUCTOR_CACHE_DIR=str(cache / 'inductor'))
    return env


def witness(device, rank, logdir):
    program = '''import json,torch
assert torch.cuda.is_available() and torch.cuda.device_count()==1
free,total=torch.cuda.mem_get_info()
assert free >= 74000*1024*1024, (free,total)
torch.manual_seed(20261016)
x=torch.randn(64,64,device='cuda',dtype=torch.float32)
y=x@x
torch.cuda.synchronize()
assert torch.isfinite(y).all()
print(json.dumps(dict(state='WITNESS_OK',gpu=torch.cuda.get_device_name(0),free_bytes=free,
 total_bytes=total,checksum=float(y.double().sum().item()),torch=torch.__version__,cuda=torch.version.cuda)))
'''
    result = subprocess.run([sys.executable, '-c', program], env=environment(device, 'witness_' + str(rank)),
                            capture_output=True, text=True, timeout=120)
    (logdir / ('witness_' + str(rank) + '.log')).write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError('GPU witness failed for visible device ' + device)
    record = json.loads(result.stdout.strip().splitlines()[-1])
    record.update(cuda_visible_device=device, rank=rank)
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--model', choices=MODEL_NAMES)
    p.add_argument('--num-shards', type=int)
    p.add_argument('--smoke-job-id')
    args = p.parse_args()
    job = os.environ['SLURM_JOB_ID']
    node_rank = int(os.environ.get('SLURM_PROCID', '0'))
    nodes = int(os.environ.get('SLURM_JOB_NUM_NODES', '1'))
    devices = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if len(devices) != 8 or any(not item for item in devices) or len(set(devices)) != 8:
        raise RuntimeError('Expected eight distinct allocated CUDA_VISIBLE_DEVICES')
    tasks = RUN / 'prepared/tasks.jsonl'
    if sha(tasks) != TASK_HASH:
        raise RuntimeError('Canonical task manifest has changed')
    logdir = RUN / 'logs' / ('job_' + job) / socket.gethostname()
    logdir.mkdir(parents=True, exist_ok=True)
    if not args.smoke:
        if args.model is None or args.num_shards != nodes * 8:
            raise RuntimeError('Shard count must match all eight GPUs per allocated node')
        passed = json.loads((RUN / 'smoke' / ('job_' + args.smoke_job_id) / 'PASSED.json').read_text())
        if passed['worker_sha256'] != sha(CODE / 'worker.py') or passed['tasks_sha256'] != TASK_HASH:
            raise RuntimeError('Startup smoke does not match deployed worker/tasks')
    with ThreadPoolExecutor(max_workers=8) as pool:
        records = list(pool.map(lambda pair: witness(pair[1], pair[0], logdir), enumerate(devices)))
    atomic(logdir / 'GPU_WITNESSES.json', {'host': socket.gethostname(), 'gpus': records})
    children = []
    handles = []

    def cleanup(*_):
        for child in children:
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + 15
        for child in children:
            try:
                child.wait(timeout=max(.1, deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        for handle in handles:
            handle.close()

    def terminate(signum, _frame):
        cleanup()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    try:
        assignments = [(i, name, 0, 1) for i, name in enumerate(MODEL_NAMES)] if args.smoke else [
            (i, args.model, node_rank*8+i, args.num_shards) for i in range(8)]
        for gpu, model, shard, total in assignments:
            output = RUN / ('smoke/job_' + job if args.smoke else 'results') / model
            cmd = [sys.executable, '-u', str(CODE/'worker.py'), '--model-path', str(MODELS/model),
                   '--tasks', str(tasks), '--output', str(output), '--shard-index', str(shard),
                   '--num-shards', str(total), '--batch-size', '1', '--gpu-memory-utilization', '.90',
                   '--max-num-batched-tokens', '8192', '--enforce-eager']
            if args.smoke:
                cmd += ['--smoke-limit', '5', '--smoke-max-tokens', '64']
            handle = (logdir / (model + '_shard' + str(shard) + '.log')).open('w')
            handles.append(handle)
            child = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT,
                                     env=environment(devices[gpu], str(gpu)), start_new_session=True)
            children.append(child)
        atomic(logdir / 'CHILDREN.json', {'job_id': job, 'host': socket.gethostname(),
               'children': [{'pid': c.pid, 'assignment': a} for c, a in zip(children, assignments)]})
        while any(child.poll() is None for child in children):
            failures = [child.returncode for child in children if child.poll() not in (None, 0)]
            if failures:
                raise RuntimeError('A replica failed; sibling processes stopped: ' + repr(failures))
            time.sleep(3)
        if any(child.returncode != 0 for child in children):
            raise RuntimeError('Inference replica failed')
        if args.smoke:
            for model in MODEL_NAMES:
                output = RUN / 'smoke' / ('job_' + job) / model
                rows = [json.loads(file.read_text()) for file in (output/'completed').glob('*.json')]
                if len(rows) != 5 or len({row['benchmark'] for row in rows}) != 5:
                    raise RuntimeError('Startup smoke must cover all five benchmarks for ' + model)
                if any(row['status'] != 'completed' or row['mode'] != 'smoke' for row in rows):
                    raise RuntimeError('Invalid startup smoke output')
            atomic(RUN/'smoke'/('job_'+job)/'PASSED.json', {'state':'PASSED', 'job_id':job,
                   'worker_sha256':sha(CODE/'worker.py'), 'tasks_sha256':TASK_HASH,
                   'models':MODEL_NAMES, 'samples_per_model':5, 'context_tokens':262144,
                   'completed_at':time.time()})
        atomic(logdir/'NODE_COMPLETE.json', {'job_id':job, 'host':socket.gethostname(), 'finished_at':time.time()})
    finally:
        cleanup()


if __name__ == '__main__':
    main()
