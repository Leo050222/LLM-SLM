#!/usr/bin/env python3
"""Coordinator backend for strong-main/weak-score prompt-repair rollouts."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import os
import sys
import time

sys.path.insert(0, '/storage/yukaichengLab/wangxin/test_eiloeo/opd_online_code/20260911_chunk64_k8_2nodes/ocrd_pipeline')
from ablation_core import Settings, FileRPC, atomic_json, digest, rollout
from reprompt import build_prompt, TEMPLATE_SHA256
from anchor_selector.core import tokenizer_fingerprint


class Backend:
    def __init__(self, rpc, models, tokenizer):
        self.rpc, self.models, self.tokenizer = rpc, models, tokenizer
        self.max_model_len = min(v['max_model_len'] for v in models.values())

    def call(self, role, **request):
        return self.rpc.call(role, dict(model_identity=digest(self.models[role]), **request))

    def generate(self, role, prefix, n, seed, banned_first=None):
        if banned_first is None:
            return self.call(role, op='generate', prefix=prefix, n=n, seed=seed)
        if n != 1:
            raise ValueError('First-token ban requires a one-token request')
        return self.call(role, op='generate', prefix=prefix, n=1, seed=seed, banned_first=banned_first)

    def build_reprompt_prefix(self, row, response, draft, position):
        return build_prompt(self.tokenizer, row, response, draft, position)

    def _top100(self, top_ids, top_lp):
        return [{'token_id': int(tid),
                 'token_text': self.tokenizer.decode([int(tid)], skip_special_tokens=False),
                 'logprob': float(lp)} for tid, lp in zip(top_ids, top_lp)]

    def candidates(self, prefix, ids, threshold):
        """Find the earliest strong-token/weak-model signed logprob gap > threshold."""
        with ThreadPoolExecutor(max_workers=2) as pool:
            strong_future = pool.submit(self.call, 'student', op='score', prefix=prefix, tokens=ids)
            weak_future = pool.submit(self.call, 'teacher', op='score', prefix=prefix, tokens=ids)
            strong, weak = strong_future.result(), weak_future.result()
        if not (len(strong['chosen']) == len(ids) == len(weak['chosen'])):
            raise ValueError('Score length mismatch')
        candidates = []
        for pos, token_id in enumerate(ids):
            strong_lp = float(strong['chosen'][pos]); weak_lp = float(weak['chosen'][pos])
            gap = strong_lp - weak_lp
            if gap > threshold:
                candidates.append(dict(
                    response_position=pos,
                    strong_token_id=int(token_id),
                    strong_logprob=strong_lp,
                    weak_logprob_on_strong_token=weak_lp,
                    signed_logprob_gap_strong_minus_weak=gap,
                    absolute_logprob_gap=abs(gap),
                    gap_direction='strong_higher',
                    threshold=float(threshold),
                    strong_top100=self._top100(strong['top_ids'][pos], strong['top_lp'][pos]),
                    weak_top100=self._top100(weak['top_ids'][pos], weak['top_lp'][pos]),
                ))
                break
        return candidates


def replica_index(dataset_index, replicas):
    if replicas < 1: raise ValueError('replicas must be positive')
    return int(dataset_index) % replicas


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--queue', type=Path, required=True)
    p.add_argument('--replicas', type=int, default=4)
    p.add_argument('--tokenizer', required=True)
    p.add_argument('--expected-rows', type=int, default=4631)
    p.add_argument('--concurrency', type=int, default=32)
    p.add_argument('--chunk-tokens', type=int, default=64)
    p.add_argument('--teacher-tokens', type=int, default=1)
    p.add_argument('--gap-threshold', type=float, default=2.)
    p.add_argument('--max-tokens', type=int, default=40264)
    p.add_argument('--seed', type=int, default=20260911)
    p.add_argument('--rpc-timeout', type=float, default=1800)
    args = p.parse_args()
    settings = Settings(args.chunk_tokens, args.teacher_tokens, args.gap_threshold,
                        args.max_tokens, args.seed, 'strong_minus_weak_gt_positive_threshold',
                        'strong_reprompt', None, True, ())
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    if len(rows) != args.expected_rows or len({r['dataset_index'] for r in rows}) != len(rows):
        raise ValueError('Input count mismatch or duplicate dataset indices')
    if args.concurrency <= 0 or not rows: raise ValueError('Positive concurrency and nonempty input required')
    for row in rows:
        if not row.get('prompt_token_ids') or not row.get('problem'): raise ValueError('Missing prompt tokens or problem')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=False)
    queues = [args.queue / f'replica_{i}' for i in range(args.replicas)]
    replica_models = [{role: json.loads((queue / role / 'model.json').read_text()) for role in ('student', 'teacher')} for queue in queues]
    models = replica_models[0]
    if any(m != models for m in replica_models): raise ValueError('Replica model identities differ')
    fingerprint = tokenizer_fingerprint(tokenizer)
    if any(v['tokenizer_sha256'] != fingerprint for v in models.values()): raise ValueError('Tokenizer mismatch')
    code_identity = digest({p.name: p.read_text() for p in Path(__file__).parent.glob('*.py')})
    manifest = dict(code_identity=code_identity, settings=asdict(settings), input_sha256=digest(rows), models=models,
                    expected_rows=len(rows), sampling=dict(temperature=0., top_p=1., top_k=-1), topk=100,
                    multiple_takeovers=True, eos_policy='stop', first_token_ban=True,
                    reprompt_template_sha256=TEMPLATE_SHA256, mode='strong_reprompt', thinking=False,
                    role_mapping=dict(student='strong_main_and_reprompt', teacher='weak_scoring'))
    manifest['execution'] = dict(replicas=args.replicas, concurrency=args.concurrency)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    import fcntl
    run_lock = (args.output_dir / 'coordinator.lock').open('a'); fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest_path = args.output_dir / 'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('Run manifest changed; choose a new output directory')
    atomic_json(manifest_path, manifest)
    backends = [Backend(FileRPC(queue, args.rpc_timeout), models, tokenizer) for queue in queues]

    def run_one(row):
        path = args.output_dir / 'traces' / f"{row['dataset_index']:05d}.json"
        checkpoint = json.loads(path.read_text()) if path.exists() else None
        backend = backends[replica_index(row['dataset_index'], args.replicas)]
        result = rollout(row, settings, backend, None, checkpoint, save=lambda state: atomic_json(path, state))
        output = dict(row, **result)
        output['response'] = tokenizer.decode(result['response_token_ids'], skip_special_tokens=True)
        atomic_json(args.output_dir / 'completed' / path.name, output)
        return row['dataset_index']

    started, failures = time.monotonic(), []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        pending = {pool.submit(run_one, row): row['dataset_index'] for row in rows}
        for future in as_completed(pending):
            try: print(json.dumps({'completed': future.result()}), flush=True)
            except Exception as error:
                failures.append(dict(dataset_index=pending[future], error_type=type(error).__name__))
                print(json.dumps({'failed': failures[-1]}), flush=True)
    atomic_json(args.output_dir / 'run_status.json', dict(expected=len(rows), failed=failures,
                completed=len(rows) - len(failures), elapsed_seconds=time.monotonic() - started))
    if failures: raise SystemExit('Some questions failed; inspect workers and rerun unchanged to resume')

if __name__ == '__main__': main()
