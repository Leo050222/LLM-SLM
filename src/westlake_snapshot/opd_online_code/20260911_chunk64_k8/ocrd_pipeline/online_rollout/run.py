#!/usr/bin/env python3
"""Networked coordinator; GPU workers communicate through a shared directory."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from online_rollout.core import Settings, FileRPC, atomic_json, digest, rollout
from anchor_selector.core import build_candidates, tokenizer_fingerprint
from anchor_selector.openlux_client import OpenLuxClient, build_user_prompt, SYSTEM_PROMPT_SHA256, load_env_file


class Backend:
    def __init__(self, rpc, models, tokenizer):
        self.rpc, self.models, self.tokenizer = rpc, models, tokenizer
        self.max_model_len = min(v['max_model_len'] for v in models.values())

    def call(self, role, **request):
        return self.rpc.call(role, dict(model_identity=digest(self.models[role]), **request))

    def generate(self, role, prefix, n, seed, banned_first=None):
        if banned_first is None:
            return self.call(role, op='generate', prefix=prefix, n=n, seed=seed)
        first = self.call(role, op='generate', prefix=prefix, n=1, seed=seed,
                          banned_first=banned_first)
        if len(first['token_ids']) != 1:
            raise ValueError('Teacher must return one entry token')
        if n == 1 or first['finish_reason'] == 'stop':
            return first
        tail = self.call(role, op='generate', prefix=prefix + first['token_ids'], n=n - 1,
                         seed=(seed + 1) % (2**31))
        return dict(token_ids=first['token_ids'] + tail['token_ids'],
                    finish_reason=tail['finish_reason'])

    def candidates(self, prefix, ids, threshold):
        import numpy as np
        # Re-score the student too: gap is between untempered model probabilities,
        # independent of generation logprob processing in the installed vLLM build.
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(self.call, role, op='score', prefix=prefix, tokens=ids)
                       for role in ('student', 'teacher')]
            student, teacher = [f.result() for f in futures]
        return build_candidates(dict(prompt_token_ids=prefix, response_token_ids=ids),
                                self.tokenizer, np.asarray(student['chosen']),
                                np.asarray(teacher['chosen']), np.asarray(teacher['top_ids']),
                                np.asarray(teacher['top_lp']), threshold)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True, help='Prompt JSONL from prepare.py')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--queue', type=Path, required=True)
    p.add_argument('--tokenizer', required=True)
    p.add_argument('--judge-config', type=Path, required=True)
    p.add_argument('--secrets', type=Path, help='Existing judge env file; never copied into artifacts')
    p.add_argument('--expected-rows', type=int, default=1000)
    p.add_argument('--concurrency', type=int, default=8)
    p.add_argument('--chunk-tokens', type=int, default=64)
    p.add_argument('--teacher-tokens', type=int, default=8)
    p.add_argument('--gap-threshold', type=float, default=2.)
    p.add_argument('--max-tokens', type=int, default=40264)
    p.add_argument('--seed', type=int, default=20260911)
    p.add_argument('--rpc-timeout', type=float, default=1800)
    args = p.parse_args()
    settings = Settings(args.chunk_tokens, args.teacher_tokens, args.gap_threshold,
                        args.max_tokens, args.seed)
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    if len(rows) != args.expected_rows or len({r['dataset_index'] for r in rows}) != len(rows):
        raise ValueError('Input count mismatch or duplicate dataset indices')
    if args.concurrency <= 0 or not rows:
        raise ValueError('Positive concurrency and nonempty input required')
    for row in rows:
        if not row.get('prompt_token_ids') or not row.get('problem'):
            raise ValueError('Missing prompt tokens or problem')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True,
                                              trust_remote_code=False)
    models = {role: json.loads((args.queue / role / 'model.json').read_text())
              for role in ('student', 'teacher')}
    fingerprint = tokenizer_fingerprint(tokenizer)
    if any(v['tokenizer_sha256'] != fingerprint for v in models.values()):
        raise ValueError('Student/teacher/coordinator tokenizer mismatch')
    config = json.loads(args.judge_config.read_text())
    secrets = load_env_file(args.secrets) if args.secrets else {}
    key = os.environ.get('OPENLUX_API_KEY') or secrets.get('OPENLUX_API_KEY', '')
    client = OpenLuxClient(config['api_url'], key, config['judge_model'],
                           temperature=config.get('judge_temperature', 0.),
                           max_tokens=config.get('judge_max_tokens', 1200),
                           timeout_seconds=config.get('judge_timeout_seconds', 180),
                           max_retries=config.get('judge_max_retries', 3),
                           reasoning_effort=config.get('judge_reasoning_effort'),
                           extra_body=config.get('judge_extra_body'))
    code_identity = digest({p.name: p.read_text() for p in Path(__file__).parent.glob('*.py')})
    manifest = dict(code_identity=code_identity, settings=asdict(settings), input_sha256=digest(rows), models=models,
                    judge_config=config, judge_prompt_sha256=SYSTEM_PROMPT_SHA256,
                    expected_rows=len(rows), sampling=dict(temperature=1., top_p=1.),
                    multiple_takeovers=True, eos_policy='stop', first_token_ban=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    import fcntl
    run_lock = (args.output_dir / 'coordinator.lock').open('a')
    fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest_path = args.output_dir / 'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('Run manifest changed; choose a new output directory')
    atomic_json(manifest_path, manifest)
    backend = Backend(FileRPC(args.queue, args.rpc_timeout), models, tokenizer)

    def judge(row, prefix, candidate):
        text = tokenizer.decode(prefix, skip_special_tokens=False)
        # Never pass solution or any future draft tokens to the judge.
        candidate = dict(candidate, response_position=len(prefix) - len(row['prompt_token_ids']))
        user_prompt = build_user_prompt({'problem': row['problem']}, candidate, text,
                                        False, len(text))
        path = args.output_dir / 'judge_cache' / (digest([manifest['judge_prompt_sha256'],
                                                       config, user_prompt]) + '.json')
        if path.exists():
            return json.loads(path.read_text())['judgment']
        judge_started = time.monotonic()
        judgment, metadata = client.judge(user_prompt)
        metadata['elapsed_seconds'] = time.monotonic() - judge_started
        atomic_json(path, dict(judgment=judgment, metadata=metadata))
        return judgment

    def run_one(row):
        path = args.output_dir / 'traces' / f"{row['dataset_index']:05d}.json"
        checkpoint = json.loads(path.read_text()) if path.exists() else None
        result = rollout(row, settings, backend, judge, checkpoint,
                         save=lambda state: atomic_json(path, state))
        output = dict(row, **result)
        output['response'] = tokenizer.decode(result['response_token_ids'], skip_special_tokens=True)
        atomic_json(args.output_dir / 'completed' / path.name, output)
        return row['dataset_index']

    started, failures = time.monotonic(), []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        pending = {pool.submit(run_one, row): row['dataset_index'] for row in rows}
        for future in as_completed(pending):
            try:
                index = future.result()
                print(json.dumps({'completed': index}), flush=True)
            except Exception as error:
                # Do not copy provider response bodies (possibly sensitive) into run logs.
                failures.append(dict(dataset_index=pending[future], error_type=type(error).__name__))
                print(json.dumps({'failed': failures[-1]}), flush=True)
    atomic_json(args.output_dir / 'run_status.json', dict(expected=len(rows), failed=failures,
                completed=len(rows) - len(failures), elapsed_seconds=time.monotonic() - started))
    if failures:
        raise SystemExit('Some questions failed; inspect workers and rerun unchanged to resume')


if __name__ == '__main__':
    main()
