"""Greedy generation with frozen prompts, immutable resume checks and receipts."""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import time
from pathlib import Path

import requests

def digest(blob):
    return hashlib.sha256(blob).hexdigest()

def atomic(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False))
    tmp.replace(path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', type=int, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output/'generation.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    reference = AutoTokenizer.from_pretrained('/storage/yukaichengLab/share/model/Qwen3-1.7B',local_files_only=True)
    assert tok.get_vocab() == reference.get_vocab(), 'Frozen token IDs require identical vocabularies'
    blob = args.input.read_bytes()
    rows = [json.loads(l) for l in blob.splitlines()]
    assert len(rows) == 2817
    # Keep the request to the OpenAI-compatible fields accepted by all vLLM
    # versions used on the cluster.  temperature=0 is the greedy setting.
    config = dict(temperature=0.0, top_p=1.0, n=1, seed=20261005)
    identity = dict(model=args.model_name, model_path=args.model_path,
                    prompt_sha256=digest(blob), generation_config=config,
                    max_response_tokens=40264, max_context=40960)
    identity['signature'] = digest(json.dumps(identity,sort_keys=True).encode())
    manifest = args.output/'manifest.json'
    if manifest.exists():
        assert json.loads(manifest.read_text()) == identity
    else:
        atomic(manifest, identity)
    dest = args.output/'completed'
    dest.mkdir(exist_ok=True)

    def one(row):
        path = dest/f"{row['dataset_index']:05d}.json"
        if path.exists():
            x = json.loads(path.read_text())
            assert x['signature'] == identity['signature'] and x['status']=='COMPLETE'
            assert x['problem_id']==row['problem_id'] and x['prompt_token_ids']==row['prompt_token_ids']
            return
        prompt = row['prompt_token_ids']
        maximum = min(40264,40960-len(prompt)-1)
        assert maximum > 0
        payload = dict(config,model=args.model_name,prompt=prompt,
                       max_tokens=maximum,stream=False,return_token_ids=True)
        with requests.Session() as session:
            response = session.post(f'http://127.0.0.1:{args.port}/v1/completions',json=payload,timeout=7200)
            response.raise_for_status()
            response = response.json()
        choice = response['choices'][0]
        assert choice['finish_reason'] in ('stop','length'), choice
        token_ids = choice.get('token_ids')
        # Some vLLM releases ignore return_token_ids for the completions API.
        # Re-tokenize the returned text so the receipt remains inspectable.
        if token_ids is None:
            token_ids = tok.encode(choice['text'], add_special_tokens=False)
        assert len(token_ids) <= maximum
        atomic(path,dict(row,status='COMPLETE',model=args.model_name,
                         signature=identity['signature'],response=choice['text'],
                         response_token_ids=token_ids,finish_reason=choice['finish_reason'],
                         usage=response.get('usage'),generated_at=time.time()))

    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=64) as pool:
        futures = {pool.submit(one,row):row for row in rows}
        for n,future in enumerate(concurrent.futures.as_completed(futures),1):
            try:
                future.result()
            except Exception as exc:
                row=futures[future]
                errors.append(dict(index=row['dataset_index'],error=repr(exc)))
                print('GENERATION_ERROR',errors[-1],flush=True)
            if n%100==0 or n==len(rows):
                print('GENERATED',n,'/',len(rows),'errors',len(errors),flush=True)
    if errors:
        atomic(args.output/'GENERATION_ERRORS.json',errors)
        raise RuntimeError('Generation failed; preserve completed outputs for resume')
    for row in rows:
        x=json.loads((dest/f"{row['dataset_index']:05d}.json").read_text())
        assert x['signature']==identity['signature'] and x['status']=='COMPLETE'
        assert x['dataset']==row['dataset'] and x['problem_id']==row['problem_id']
    atomic(args.output/'ALL_GENERATION_COMPLETE.json',dict(rows=len(rows),signature=identity['signature'],completed_at=time.time()))

if __name__=='__main__':
    main()
