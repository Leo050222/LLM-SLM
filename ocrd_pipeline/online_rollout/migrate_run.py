#!/usr/bin/env python3
"""Snapshot a stopped single-pair run into an eight-pair run using hard links."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from online_rollout.core import atomic_json, digest, Settings
from dataclasses import asdict


def link_jsons(source, target):
    target.mkdir(parents=True, exist_ok=True)
    paths=sorted(source.glob('*.json'))
    for path in paths:
        os.link(path, target/path.name)
    return len(paths)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--target',type=Path,required=True)
    args=p.parse_args()
    if args.target.exists():
        raise ValueError('Target must be a new run directory')
    source=args.source/'results'
    lock=(source/'coordinator.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not (args.source/'queue/STOP').exists():
        raise ValueError('Old workers must be stopped before snapshot')
    manifest=json.loads((source/'manifest.json').read_text())
    assert manifest['settings']==asdict(Settings())
    assert manifest['expected_rows']==1000 and manifest['multiple_takeovers']
    assert manifest['models']['student']['worker_sha256']==digest((Path(__file__).parent/'worker.py').read_text())
    rows=[json.loads(line) for line in (args.source/'prompts1000.jsonl').read_text().splitlines()]
    assert digest(rows)==manifest['input_sha256']
    by_id={row['dataset_index']:row for row in rows}
    states=[json.loads(p.read_text()) for p in (source/'traces').glob('*.json')]
    for state in states:
        assert len(state['response_token_ids'])==len(state['token_sources'])
    for path in (source/'traces').glob('*.json'):
        state=json.loads(path.read_text()); row=by_id[int(path.stem)]
        assert state['identity']==digest({'row':row,'settings':manifest['settings']})
    args.target.mkdir(parents=True)
    os.link(args.source/'prompts1000.jsonl',args.target/'prompts1000.jsonl')
    os.link(args.source/'prompts1000.manifest.json',args.target/'prompts1000.manifest.json')
    counts={name:link_jsons(source/name,args.target/'results'/name)
            for name in ('traces','completed','judge_cache')}
    # Preserve completed GPU calls, including work after the last committed checkpoint.
    # Result files are immutable/atomically replaced, so hard links cannot mutate source.
    for role in ('student','teacher'):
        result_source=args.source/'queue'/role/'results'
        for path in result_source.glob('*.json'):
            if 'error' in json.loads(path.read_text()):
                raise ValueError('Source contains cached GPU errors')
        for replica in range(8):
            link_jsons(result_source,args.target/'queue'/f'replica_{replica}'/role/'results')
    migration=dict(source_run=str(args.source),source_manifest=manifest,counts=counts,
                   replicas=8,concurrency=256,takeover_count_limit=None)
    atomic_json(args.target/'results/migration.json',migration)
    print(json.dumps(counts))


if __name__=='__main__':
    main()
