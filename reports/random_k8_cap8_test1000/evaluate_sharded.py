#!/usr/bin/env python3
"""Score complete runs using the existing anchor comparison's boolean-normalized scorer."""
import argparse
import json
from pathlib import Path
import sys
import os
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.environ['OPD_CODE_ROOT'] + '/ocrd_pipeline')
from online_rollout.core import atomic_json, digest
from full_rollout.analyze_teacher_rollout_shard import score_math_response


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--timeout', type=float, default=20.)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--shard-count', type=int, default=1)
    args = p.parse_args()
    manifest = json.loads((args.run_dir / 'manifest.json').read_text())
    rows = [json.loads(p.read_text()) for p in sorted((args.run_dir / 'completed').glob('*.json'))]
    if len(rows) != manifest['expected_rows'] or len({r['dataset_index'] for r in rows}) != len(rows):
        raise ValueError('Incomplete run: all test questions must be included')
    rows = [r for r in rows if r['dataset_index'] % args.shard_count == args.shard_index]
    def score_one(row):
        if row['status'] != 'COMPLETE' or len(row['response_token_ids']) != len(row['token_sources']):
            raise ValueError('Invalid completed trace')
        path = args.run_dir / 'scores' / f"{row['dataset_index']:05d}.json"
        key = digest([row['response'], row['solution'], args.timeout,
                      Path(sys.modules[score_math_response.__module__].__file__).read_text()])
        cached = json.loads(path.read_text()) if path.exists() else None
        if cached and cached['identity'] == key:
            return cached
        correct, predicted, method = score_math_response(row['response'], row['solution'], args.timeout)
        score = dict(identity=key, dataset_index=row['dataset_index'], correct=correct,
                     predicted=predicted, scoring_method=method,
                     takeovers=sum('bridge_token_ids' in e for e in row['events']),
                     teacher_tokens=row['token_sources'].count('teacher'),
                     student_draft_tokens=row['student_draft_tokens'],
                     discarded_tokens=row['discarded_tokens'], judge_calls=row['judge_calls'])
        atomic_json(path, score)
        return score
    # Load reward dependencies once before concurrent calls. Symbolic checks use
    # independent spawn processes with the unchanged per-question timeout.
    from verl.utils.reward_score.math_reward import strip_string
    strip_string("1")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        scores = list(pool.map(score_one, rows))
    if args.shard_count > 1:
        print(json.dumps(dict(shard=args.shard_index, scored=len(scores))), flush=True)
        return
    totals = {k: sum(r[k] for r in scores) for k in ('correct', 'takeovers', 'teacher_tokens',
              'student_draft_tokens', 'discarded_tokens', 'judge_calls')}
    summary = dict(questions=len(rows), accuracy=totals['correct'] / len(rows), **totals,
                   scoring='existing anchor boolean-normalized scorer', rollouts_per_question=1)
    atomic_json(args.run_dir / 'summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
