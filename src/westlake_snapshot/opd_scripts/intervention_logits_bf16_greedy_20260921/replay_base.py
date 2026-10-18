"""Frozen-prefix, full-vocabulary BF16 logits archive; no stochastic generation."""
import argparse
import collections
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import struct
import time

ROOT = Path('/storage/yukaichengLab/wangxin/test_eiloeo')
OUT = ROOT/'opd_analysis/intervention_logits_bf16_x8_20260920'
MODELS = {r: Path('/storage/yukaichengLab/share/model')/('Qwen3-'+s)
          for r, s in [('student', '1.7B'), ('teacher', '14B')]}
PROTOCOL = 'accepted_prefix_raw_bf16_sdpa_chunk512_v1'
VOCAB = 151936


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def encoded(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    with tmp.open('wb') as f:
        f.write(encoded(obj)+b'\n'); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def file_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    f = path.open('a')
    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return f


def freeze_trace(x, source, replicate, family, event_start):
    assert x['status'] == 'COMPLETE'
    prompt, response = x['prompt_token_ids'], x['response_token_ids']
    assert prompt and len(response) == len(x['token_sources'])
    ids = prompt + response
    events = []
    prefix_hash = hashlib.sha256()
    hashed = 0
    previous = -1
    for ei, e in enumerate(x['events']):
        if 'bridge_token_ids' not in e:
            continue
        j = e['anchor_response_position']; c = e['selected_candidate']
        assert previous < j < len(response)
        assert len(e['bridge_token_ids']) == 1 and e['first_token_ban'] is True
        y = e['bridge_token_ids'][0]; token_x = e['student_entry_token']
        assert response[j] == y and x['token_sources'][j] == 'teacher' and y != token_x
        assert c['student_token_id'] == token_x and c['signed_logprob_gap_student_minus_teacher'] > 2
        n = len(prompt)+j
        assert c['absolute_sequence_position'] == n
        prefix_hash.update(struct.pack('<'+'I'*(n-hashed), *ids[hashed:n]))
        hashed = n; previous = j
        events.append(dict(event_id=event_start+len(events), event_index=ei,
                           response_position=j, prefix_length=n, hidden_position=n-1,
                           prefix_ids_u32le_sha256=prefix_hash.hexdigest(),
                           student_token_id=token_x, teacher_token_id=y,
                           historical_student_logprob=c['student_logprob'],
                           historical_teacher_logprob=c['teacher_logprob_on_student_token'],
                           historical_teacher_top_ids=[t['token_id'] for t in c['teacher_top_tokens']],
                           historical_teacher_top_logprobs=[t['logprob'] for t in c['teacher_top_tokens']]))
    assert len(events) == x['token_sources'].count('teacher')
    return dict(source=str(source), source_sha256=sha(source), family=family,
                dataset=x['dataset'], problem_id=x['problem_id'], dataset_index=x['dataset_index'],
                replicate=replicate, finish_reason=x['finish_reason'], prompt_length=len(prompt),
                response_length=len(response), input_ids=ids[:events[-1]['prefix_length']] if events else [],
                events=events)


def prepare(out):
    lock = file_lock(out/'prepare.lock')
    if (out/'INPUTS_COMPLETE.json').exists():
        print(json.dumps(read(out/'INPUTS_COMPLETE.json')), flush=True); return
    if shutil.disk_usage(out).free < 350_000_000_000:
        raise RuntimeError('Need at least 350 GB free before starting the full archive')
    inputs = out/'inputs'; inputs.mkdir(exist_ok=True)
    shards = []; pending = []; ne = 0; ntokens = 0
    count = 0; event_count = 0; total_tokens = 0; zero = 0
    groups = []; ds_counts = collections.defaultdict(collections.Counter)
    smoke = {}; longest = None; noevents = []

    def flush():
        nonlocal pending, ne, ntokens
        if not pending:
            return
        name = f'{len(shards):05d}.jsonl'; p = inputs/name
        tmp = p.with_suffix('.tmp')
        with tmp.open('wb') as f:
            for row in pending:
                f.write(encoded(row)+b'\n')
            f.flush(); os.fsync(f.fileno())
        tmp.replace(p)
        shards.append(dict(name=name, sha256=sha(p), traces=len(pending), events=ne,
                           input_tokens=ntokens, first_event_id=pending[0]['events'][0]['event_id']))
        pending = []; ne = 0; ntokens = 0

    for family, rel in [('code', 'opd_eval_code/code_grid_x8_20260916'),
                        ('math', 'opd_eval_hard/math_reasoning_grid_x8_20260916')]:
        root = ROOT/rel
        old = next(c for c in read(root/'replicate0_audit.json') if c['config'] == 'k1_t2')
        sources = [(0, Path(old['path']))] + [(c['replicate_index'], root/c['name'])
                    for c in read(root/'configs.json') if c['base_config'] == 'k1_t2']
        assert sorted(r for r, _ in sources) == list(range(8))
        for rep, group in sources:
            m = read(group/'manifest.json')
            assert m['settings']['teacher_tokens'] == 1 and m['settings']['gap_threshold'] == 2
            assert m['thinking'] is False and m['first_token_ban'] is True
            assert m['sampling'] == dict(temperature=1., top_p=1.)
            for role in MODELS:
                assert m['models'][role]['model'] == str(MODELS[role])
            files = sorted((group/'completed').glob('*.json'))
            assert len(files) == (2317 if family == 'code' else 2254)
            groups.append(dict(family=family, replicate=rep, source=str(group),
                               manifest_sha256=sha(group/'manifest.json'), manifest=m))
            for p in files:
                x = read(p)
                row = freeze_trace(x, p, rep, family, event_count)
                count += 1; n = len(row['events']); event_count += n
                ds_counts[row['dataset']]['traces'] += 1
                ds_counts[row['dataset']]['events'] += n
                if not n:
                    zero += 1
                    noevents.append({k:v for k,v in row.items() if k not in ['input_ids', 'events']})
                    continue
                key = f'{family}/{rep}/{row["dataset"]}'
                if key not in smoke:
                    # First real event from each dataset/replicate, with its exact prefix.
                    s = dict(row); s['events'] = row['events'][:1]
                    s['input_ids'] = row['input_ids'][:s['events'][0]['prefix_length']]
                    smoke[key] = s
                if longest is None or len(row['input_ids']) > len(longest['input_ids']):
                    longest = row
                if pending and (ne+n > 1024 or ntokens+len(row['input_ids']) > 250000):
                    flush()
                pending.append(row); ne += n; ntokens += len(row['input_ids'])
                total_tokens += len(row['input_ids'])
            flush()
            print('FROZEN_GROUP', family, rep, 'traces', count, 'events', event_count, flush=True)
    assert count == 36568 and event_count == 399601
    assert zero == 816
    atomic_json(out/'no_intervention_traces.json', noevents)
    atomic_json(out/'smoke_inputs.json', list(smoke.values()))
    atomic_json(out/'longest_trace.json', longest)
    manifest = dict(protocol=PROTOCOL, dtype='BF16', vocabulary_size=VOCAB, models={k:str(v) for k,v in MODELS.items()},
                    traces=count, events=event_count, no_intervention_traces=zero,
                    forward_input_tokens_per_model=total_tokens, tensor_bytes_two_models=event_count*VOCAB*4,
                    groups=groups, datasets=dict(ds_counts), shards=shards,
                    smoke_sha256=sha(out/'smoke_inputs.json'), longest_sha256=sha(out/'longest_trace.json'),
                    no_intervention_sha256=sha(out/'no_intervention_traces.json'))
    atomic_json(out/'manifest.json', manifest)
    atomic_json(out/'INPUTS_COMPLETE.json', dict(manifest_sha256=sha(out/'manifest.json'),
                traces=count, events=event_count, shards=len(shards), input_tokens_per_model=total_tokens,
                tensor_bytes_two_models=event_count*VOCAB*4, created=time.time()))
    print(json.dumps(read(out/'INPUTS_COMPLETE.json')), flush=True)


def load_manifest(out):
    receipt = read(out/'INPUTS_COMPLETE.json')
    assert sha(out/'manifest.json') == receipt['manifest_sha256']
    m = read(out/'manifest.json'); assert m['protocol'] == PROTOCOL
    return m, receipt['manifest_sha256']


def chunk_selected(ids_length, events, chunk_size=512):
    """Return input slices and event hidden-state indices; includes chunk boundaries."""
    cursor = 0
    for start in range(0, ids_length, chunk_size):
        end = min(ids_length, start+chunk_size)
        selected = []
        while cursor < len(events) and events[cursor]['hidden_position'] < end:
            pos = events[cursor]['hidden_position']
            assert start <= pos < end
            selected.append((cursor, pos-start)); cursor += 1
        yield start, end, selected
    assert cursor == len(events)


def replay(model, row):
    import torch
    cache = None; result = []
    ids = row['input_ids']
    for start, end, selected in chunk_selected(len(ids), row['events']):
        step = model.model(input_ids=torch.tensor([ids[start:end]], device='cuda'),
                           attention_mask=torch.ones((1, end), dtype=torch.long, device='cuda'),
                           past_key_values=cache, use_cache=True, return_dict=True)
        cache = step.past_key_values
        assert cache.get_seq_length() == end
        if selected:
            h = step.last_hidden_state[0, [j for _, j in selected]]
            logits = model.lm_head(h)
            assert logits.dtype == torch.bfloat16 and logits.shape == (len(selected), VOCAB)
            assert torch.isfinite(logits).all()
            result.append(logits.detach().cpu())
            del logits, h
        del step
    del cache
    values = torch.cat(result)
    assert values.shape == (len(row['events']), VOCAB)
    return values


def run_smoke(model, role, out, m, manifest_hash, runtime_hash):
    import torch
    from safetensors.torch import save_file, load_file
    assert sha(out/'smoke_inputs.json') == m['smoke_sha256']
    assert sha(out/'longest_trace.json') == m['longest_sha256']
    rows = read(out/'smoke_inputs.json')
    checks = []; tensors = []
    for row in rows:
        v = replay(model, row)
        e = row['events'][0]; lp = v[0].float().log_softmax(-1)
        old = e[f'historical_{role}_logprob']; new = lp[e['student_token_id']].item()
        checks.append(dict(dataset=row['dataset'], replicate=row['replicate'], event_id=e['event_id'],
                           prefix_length=e['prefix_length'], historical_logprob=old,
                           replay_logprob=new, abs_logprob_error=abs(old-new)))
        tensors.append(v)
    # Validate causal indexing for adjacent and separated actual events with independent prefills.
    row = read(out/'longest_trace.json')
    # Exercise the real longest context and all its event projections (including EOS if present).
    long_values = replay(model, row)
    assert long_values.shape[0] == len(row['events'])
    controls = []
    # Cap fresh-prefill controls to 4096 tokens; long context memory is exercised above.
    chosen = [i for i,e in enumerate(row['events']) if e['prefix_length'] <= 4096]
    chosen = sorted(set(([chosen[0], chosen[len(chosen)//2], chosen[-1]] if chosen else [])))
    if not chosen:
        # Use a first-event fixture if the longest trace's first intervention is late.
        row = min(rows, key=lambda x: len(x['input_ids']))
        long_values = replay(model, row); chosen = [0]
    for i in chosen:
        e = row['events'][i]; prefix = row['input_ids'][:e['prefix_length']]
        h = model.model(input_ids=torch.tensor([prefix], device='cuda'), use_cache=False,
                        return_dict=True).last_hidden_state[0, -1:]
        direct = model.lm_head(h).detach().cpu()[0]
        p = long_values[i].float().softmax(-1); q = direct.float().softmax(-1)
        tv = (p-q).abs().sum().item()/2
        controls.append(dict(event_id=e['event_id'], prefix_length=len(prefix),
                             total_variation=tv, argmax_equal=p.argmax().item() == q.argmax().item()))
        assert tv < .1, ('chunk/full alignment numerical gate', controls[-1])
        del h, direct
    # Dtype/bitwise round-trip proves the archive does not silently convert to FP16/FP32.
    target = out/role/'smoke.safetensors'
    saved = torch.cat(tensors).contiguous()
    save_file({'logits': saved}, str(target), metadata={'dtype':'BF16', 'protocol':PROTOCOL})
    loaded = load_file(str(target))['logits']
    assert loaded.dtype == torch.bfloat16 and torch.equal(saved.view(torch.int16), loaded.view(torch.int16))
    errors = sorted(c['abs_logprob_error'] for c in checks)
    # Wide stop gate catches gross alignment/model errors without claiming exact replay.
    assert errors[len(errors)//2] < 1 and max(errors) < 5, ('historical calibration failed', errors)
    atomic_json(out/role/'SMOKE_COMPLETE.json', dict(manifest_sha256=manifest_hash, runtime_sha256=runtime_hash,
                historical_calibration=checks, logprob_error_median=errors[len(errors)//2],
                logprob_error_max=max(errors), alignment_controls=controls,
                longest_prefix_tested=len(read(out/'longest_trace.json')['input_ids']),
                bitwise_bf16_roundtrip=True, completed=time.time()))
    print('SMOKE_COMPLETE', role, 'calibration', len(checks), 'median/max', errors[len(errors)//2], max(errors), flush=True)


def validate_shard(out, role, shard, manifest_hash, runtime_hash, checksum=True):
    from safetensors import safe_open
    base = out/role/Path(shard['name']).stem
    receipt = read(base.with_suffix('.json'))
    assert receipt['manifest_sha256'] == manifest_hash and receipt['runtime_sha256'] == runtime_hash
    assert receipt['input_sha256'] == shard['sha256'] and receipt['events'] == shard['events']
    p = base.with_suffix('.safetensors')
    assert p.stat().st_size == receipt['bytes']
    if checksum:
        assert sha(p) == receipt['sha256']
    with safe_open(str(p), framework='pt', device='cpu') as f:
        assert f.get_slice('logits').get_shape() == [shard['events'], VOCAB]
        assert f.get_slice('logits').get_dtype() == 'BF16'
        assert f.metadata()['manifest_sha256'] == manifest_hash
    return receipt


def score(out, role):
    # gnho008 CUDA ordinals differ from nvidia-smi/Slurm physical IDs.
    # Bind only the GPU allocated to this job, using its stable UUID.
    if os.environ.get('SLURMD_NODENAME') == 'gnho008' and os.environ.get('SLURM_GPUS_ON_NODE') == '1':
        import csv
        import subprocess
        allocated = os.environ['SLURM_JOB_GPUS'].split(',')
        assert len(allocated) == 1 and allocated[0].isdigit(), allocated
        inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv'], text=True)
        mapping = {r['index']: r['uuid'] for r in csv.DictReader(inventory.splitlines(), skipinitialspace=True)}
        # Device cgroups can expose just the allocated card and renumber it to 0.
        uuid = next(iter(mapping.values())) if len(mapping) == 1 else mapping[allocated[0]]
        assert uuid != 'GPU-c33c45c6-61d6-36bc-e5ac-6d80e3bc4968', 'Slurm assigned unhealthy physical GPU2'
        os.environ['CUDA_VISIBLE_DEVICES'] = uuid
        print('SLURM_GPU_BINDING', allocated[0], uuid, flush=True)
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from safetensors.torch import save_file
    m, manifest_hash = load_manifest(out)
    lock = file_lock(out/role/'worker.lock')
    torch.set_num_threads(8)
    torch.manual_seed(20260920); torch.cuda.manual_seed_all(20260920)
    torch.backends.cuda.matmul.allow_tf32 = False
    free, total = torch.cuda.mem_get_info()
    assert free > 70*1024**3, ('Allocated GPU is not free enough', free, total)
    x = torch.randn(256,256,device='cuda'); y = x@x
    assert torch.isfinite(y).all()
    print('CUDA_WITNESS_OK', torch.cuda.get_device_name(), y.sum().item(), flush=True)
    del x, y
    tokenizers = [AutoTokenizer.from_pretrained(p, local_files_only=True) for p in MODELS.values()]
    assert tokenizers[0].get_vocab() == tokenizers[1].get_vocab()
    token_hash = hashlib.sha256(encoded(tokenizers[0].get_vocab())).hexdigest()
    runtime = dict(protocol=PROTOCOL, model=str(MODELS[role]), dtype='bfloat16', backend='transformers_sdpa',
                   torch=torch.__version__, transformers=transformers.__version__, cuda=torch.version.cuda,
                   tokenizer_vocab_sha256=token_hash, vocab_size=VOCAB, chunk_tokens=512,
                   model_config_sha256=sha(MODELS[role]/'config.json'),
                   model_index_sha256=sha(MODELS[role]/'model.safetensors.index.json'),
                   weights=[dict(name=p.name,bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
                            for p in sorted(MODELS[role].glob('*.safetensors'))],
                   script_sha256=sha(__file__))
    runtime_hash = hashlib.sha256(encoded(runtime)).hexdigest()
    rp = out/role/'runtime.json'
    if rp.exists():
        assert read(rp) == runtime, 'Runtime changed; do not mix incompatible logits in this archive'
    else:
        atomic_json(rp, runtime)
    atomic_json(out/role/'launch.json', dict(job_id=os.environ.get('SLURM_JOB_ID'), node=platform.node(),
                gpu=torch.cuda.get_device_name(), runtime_sha256=runtime_hash, started=time.time()))
    model = AutoModelForCausalLM.from_pretrained(MODELS[role], local_files_only=True,
                   dtype=torch.bfloat16, attn_implementation='sdpa').cuda().eval()
    model.requires_grad_(False)
    assert model.config.vocab_size == VOCAB
    with torch.inference_mode():
        run_smoke(model, role, out, m, manifest_hash, runtime_hash)
        started = time.monotonic(); done = 0; new_events = 0; new_tokens = 0
        for si, shard in enumerate(m['shards']):
            base = out/role/Path(shard['name']).stem
            if base.with_suffix('.json').exists():
                validate_shard(out, role, shard, manifest_hash, runtime_hash)
                done += shard['events']; continue
            inp = out/'inputs'/shard['name']
            assert sha(inp) == shard['sha256']
            values = []; event_ids = []
            t0 = time.monotonic()
            for line in inp.read_text().splitlines():
                row = json.loads(line)
                assert sha(row['source']) == row['source_sha256'], 'Historical rollout changed'
                values.append(replay(model, row))
                event_ids.extend(e['event_id'] for e in row['events'])
            assert event_ids == list(range(shard['first_event_id'], shard['first_event_id']+shard['events']))
            tensor = torch.cat(values).contiguous(); del values
            target = base.with_suffix('.safetensors'); tmp = target.with_suffix('.tmp')
            save_file({'logits': tensor, 'event_ids': torch.tensor(event_ids,dtype=torch.int64)}, str(tmp),
                      metadata={'manifest_sha256':manifest_hash, 'runtime_sha256':runtime_hash,
                                'input_sha256':shard['sha256'], 'role':role, 'protocol':PROTOCOL,
                                'semantics':'raw pre-softmax, unmasked, accepted historical prefix'})
            with tmp.open('rb') as f:
                os.fsync(f.fileno())
            tmp.replace(target)
            receipt = dict(manifest_sha256=manifest_hash, runtime_sha256=runtime_hash,
                           input_sha256=shard['sha256'], events=shard['events'],
                           sha256=sha(target), bytes=target.stat().st_size,
                           seconds=time.monotonic()-t0, completed=time.time())
            atomic_json(base.with_suffix('.json'), receipt)
            del tensor
            validate_shard(out, role, shard, manifest_hash, runtime_hash, checksum=False)
            done += shard['events']; new_events += shard['events']; new_tokens += shard['input_tokens']
            progress = dict(role=role, events_done=done, events_total=m['events'], shards_done=si+1,
                            shards_total=len(m['shards']), elapsed_seconds=time.monotonic()-started,
                            new_events=new_events, new_input_tokens=new_tokens, updated=time.time())
            atomic_json(out/role/'PROGRESS.json', progress)
            print('SHARD_COMPLETE', json.dumps(progress), flush=True)
    atomic_json(out/role/'COMPLETE.json', dict(manifest_sha256=manifest_hash, runtime_sha256=runtime_hash,
                events=done, shards=len(m['shards']), completed=time.time()))


def verify(out):
    from safetensors import safe_open
    m, manifest_hash = load_manifest(out)
    totals = {}
    for role in MODELS:
        complete = read(out/role/'COMPLETE.json')
        assert complete['manifest_sha256'] == manifest_hash and complete['events'] == m['events']
        runtime_hash = hashlib.sha256(encoded(read(out/role/'runtime.json'))).hexdigest()
        assert runtime_hash == complete['runtime_sha256']
        n = 0; size = 0
        for shard in m['shards']:
            assert sha(out/'inputs'/shard['name']) == shard['sha256']
            receipt = validate_shard(out, role, shard, manifest_hash, runtime_hash)
            p = out/role/(Path(shard['name']).stem+'.safetensors')
            with safe_open(str(p), framework='pt', device='cpu') as f:
                assert f.get_tensor('event_ids').tolist() == list(range(n,n+shard['events']))
            n += shard['events']; size += receipt['bytes']
        assert n == m['events']
        totals[role] = dict(events=n, bytes=size, shards=len(m['shards']))
    atomic_json(out/'ALL_LOGITS_VERIFIED.json', dict(manifest_sha256=manifest_hash, roles=totals,
                dtype='BF16', vocabulary_size=VOCAB, checksum_all_files=True, completed=time.time()))
    print(json.dumps(read(out/'ALL_LOGITS_VERIFIED.json')), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare','score','verify'])
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--role', choices=list(MODELS))
    args = parser.parse_args()
    if args.command == 'prepare': prepare(args.out)
    elif args.command == 'verify': verify(args.out)
    else:
        assert args.role
        score(args.out, args.role)
