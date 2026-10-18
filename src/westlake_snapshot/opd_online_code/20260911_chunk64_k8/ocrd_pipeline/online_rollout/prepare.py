#!/usr/bin/env python3
"""Freeze all 1000 test prompts; no filtering by correctness or anchor availability."""
import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from online_rollout.core import atomic_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True, help='Original/flat parquet, or baseline rollout JSONL')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tokenizer', required=True)
    p.add_argument('--expected-rows', type=int, default=1000)
    args = p.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=False)
    if args.input.suffix == '.parquet':
        import pyarrow.parquet as pq
        source = pq.read_table(args.input).to_pylist()
    else:
        source = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    if len(source) != args.expected_rows:
        raise ValueError('Input count mismatch; do not use the selected-anchor subset')
    rows = []
    vocab_size = len(tokenizer)
    for i, row in enumerate(source):
        if 'extra_info' in row:
            row = dict(dataset_index=i, problem_id=str(row['extra_info']['problem_id']),
                       problem=row['extra_info']['question'], solution=row['reward_model']['ground_truth'])
        problem = str(row['problem'])
        ids = row.get('prompt_token_ids')
        if ids is None:
            message = ('Solve the following math problem. Show your reasoning and put your '
                       'final answer inside \\boxed{}.\n\n' + problem)
            ids = tokenizer.apply_chat_template([{'role': 'user', 'content': message}],
                                               tokenize=True, add_generation_prompt=True,
                                               enable_thinking=False)
            if isinstance(ids, Mapping):
                ids = ids['input_ids']
        if not ids or not all(isinstance(t, int) and 0 <= t < vocab_size for t in ids):
            raise ValueError('Invalid prompt token IDs')
        rows.append(dict(dataset_index=int(row.get('dataset_index', i)),
                         problem_id=str(row.get('problem_id', row.get('id', i))),
                         problem=problem, solution=str(row['solution']), prompt_token_ids=list(ids)))
    rows.sort(key=lambda r: r['dataset_index'])
    if [r['dataset_index'] for r in rows] != list(range(args.expected_rows)):
        raise ValueError('Expected contiguous unique test dataset indices')
    if len({r['problem_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate problem IDs')
    content = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows)
    if args.output.exists() and args.output.read_text() != content:
        raise ValueError('Refusing to replace different prepared prompts')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix('.tmp')
    temp.write_text(content)
    temp.replace(args.output)
    atomic_json(args.output.with_suffix('.manifest.json'),
                dict(rows=len(rows), source=str(args.input),
                     source_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                     output_sha256=hashlib.sha256(content.encode()).hexdigest(), tokenizer=args.tokenizer))


if __name__ == '__main__':
    main()
