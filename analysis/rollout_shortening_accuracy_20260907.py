#!/usr/bin/env python3
"""Compare each after-bridge rollout with its prefix's eight-rollout mean.

Reads score files only. Writes aggregate CSV/JSON/Markdown into --output.
No inference or GPU work is performed.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np


SUFFIX = Path('analysis_correctness_length_single_job/scores.jsonl')


def rows(path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def score_files(root, h, rep):
    name = f'student_after_bridge_qwen3_1_7b_h{h}_n8_x8_seed20260826'
    path = root / name / f'replicate_{rep:02d}' / SUFFIX
    if h == 256 and rep == 0:
        path = root / 'student_after_bridge_qwen3_1_7b_h256_n8_seed20260826' / SUFFIX
    if path.exists():
        return [path]
    shards = sorted((path.parent / 'shards').glob('shard_*/scores.jsonl'))
    if len(shards) != 64:
        raise ValueError(f'Missing merged file or 64 shards: {path}')
    return shards


def within_prefix_clustered_association(x, y, mask):
    """Prefix fixed-effect slope with prefix-clustered normal-approximation SE."""
    mask = np.asarray(mask, dtype=bool)
    y = np.asarray(y, dtype=float)
    counts = mask.sum(axis=(1, 2), keepdims=True)
    active = counts[:, 0, 0] > 1
    xmean = (x * mask).sum(axis=(1, 2), keepdims=True) / np.maximum(counts, 1)
    ymean = (y * mask).sum(axis=(1, 2), keepdims=True) / np.maximum(counts, 1)
    xr = np.where(mask, x - xmean, 0.0)
    yr = np.where(mask, y - ymean, 0.0)
    denominator = float((xr * xr).sum())
    beta = float((xr * yr).sum() / denominator)
    residual = np.where(mask, yr - beta * xr, 0.0)
    cluster_scores = (xr * residual).sum(axis=(1, 2))[active]
    clusters = int(active.sum())
    variance = float((clusters / (clusters - 1)) * np.square(cluster_scores).sum() / denominator**2)
    standard_error = math.sqrt(variance)
    z_value = beta / standard_error
    p_value = math.erfc(abs(z_value) / math.sqrt(2.0))
    return {
        "eligible_rollouts": int(mask.sum()),
        "prefix_clusters": clusters,
        "accuracy_pp_per_10pct_more_shortening": beta * 1000.0,
        "cluster_se_pp_per_10pct": standard_error * 1000.0,
        "cluster_z": z_value,
        "cluster_normal_p_two_sided": p_value,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/storage/yukaichengLab/wangxin/test_eiloeo/dataset/train'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-prefixes', type=int, default=13244)
    args = parser.parse_args()
    baseline_path = args.root / 'student_from_anchor_prefix_qwen3_1_7b_n8_seed20260827' / SUFFIX
    baseline = {}
    sources = [baseline_path]
    for row in rows(baseline_path):
        key, j = int(row['dataset_index']), int(row['rollout_index'])
        group = baseline.setdefault(key, {})
        assert j in range(8) and j not in group, (key, j)
        group[j] = row
    ids = sorted(baseline)
    assert len(ids) == args.expected_prefixes
    position = {key: i for i, key in enumerate(ids)}
    n = len(ids)
    base_length, base_acc, base_clean = [], [], []
    problem_ids = []
    for key in ids:
        group = baseline[key]
        assert set(group) == set(range(8))
        rr = list(group.values())
        assert len({str(r['problem_id']) for r in rr}) == 1
        problem_ids.append(str(rr[0]['problem_id']))
        base_length.append(np.mean([r['full_trace_token_count'] for r in rr]))
        base_acc.append(np.mean([bool(r['correct']) for r in rr]))
        base_clean.append(all(r['finish_reason'] != 'length' for r in rr))
    base_length, base_acc = np.array(base_length), np.array(base_acc)
    base_clean = np.array(base_clean)
    assert (base_length > 0).all()
    tables, diagnostics = [], {}
    for h in (32, 64, 128, 256):
        length = np.empty((n, 8, 8), dtype=np.int64)
        correct = np.empty((n, 8, 8), dtype=bool)
        stop = np.empty((n, 8, 8), dtype=bool)
        for rep in range(8):
            seen = np.zeros((n, 8), dtype=bool)
            files = score_files(args.root, h, rep)
            sources.extend(files)
            for path in files:
                for row in rows(path):
                    i, j = position[int(row['dataset_index'])], int(row['rollout_index'])
                    assert j in range(8) and not seen[i, j]
                    assert int(row['source_bridge_token_count']) == h
                    assert str(row['problem_id']) == problem_ids[i]
                    seen[i, j] = True
                    length[i, j, rep] = int(row['full_trace_token_count'])
                    correct[i, j, rep] = bool(row['correct'])
                    stop[i, j, rep] = row['finish_reason'] != 'length'
            assert seen.all(), (h, rep)
        delta = base_length[:, None, None] - length
        percent = 100 * delta / base_length[:, None, None]
        base = np.broadcast_to(base_acc[:, None, None], length.shape)
        masks = {'all': np.ones(length.shape, dtype=bool),
                 'clean': base_clean[:, None, None] & stop}
        diagnostics[str(h)] = {'rollouts': int(length.size), 'accuracy': float(correct.mean())}
        for scope, eligible in masks.items():
            def add(kind, label, mask):
                mask = mask & eligible
                count = int(mask.sum())
                if not count:
                    return
                tables.append(dict(horizon=h, scope=scope, grouping=kind, bucket=label,
                                   rollouts=count, correct=int(correct[mask].sum()),
                                   accuracy_pct=float(correct[mask].mean()*100),
                                   matched_baseline_accuracy_pct=float(base[mask].mean()*100),
                                   delta_vs_matched_baseline_pp=float((correct[mask]-base[mask]).mean()*100),
                                   mean_reduction_tokens=float(delta[mask].mean()),
                                   mean_reduction_pct=float(percent[mask].mean()),
                                   prefixes=int(mask.any(axis=(1, 2)).sum())))
            add('overview', 'all', eligible)
            add('overview', 'shorter', delta > 0)
            add('overview', 'not_shorter', delta <= 0)
            for kind, values, edges in (
                ('relative', percent, [0, 10, 20, 30, 40, 50, 75, float('inf')]),
                ('absolute', delta, [0, 64, 128, 256, 512, 1024, float('inf')])):
                for low, high in zip(edges[:-1], edges[1:]):
                    add(kind, f'({low},{high}]', (values > low) & (values <= high))
            positive = percent[eligible & (delta > 0)]
            if len(positive):
                edges = np.quantile(positive, np.linspace(0, 1, 6))
                for q in range(5):
                    mask = (percent >= edges[q]) & (percent < edges[q+1])
                    if q == 4:
                        mask = (percent >= edges[q]) & (percent <= edges[q+1])
                    add('shorter_quintile', f'Q{q+1}', mask & (delta > 0))
            # Within-prefix association removes between-prefix level differences.
            # Descriptive linear probability slope; not a causal estimate or CI.
            mask = eligible & (delta > 0)
            counts = mask.sum(axis=(1, 2), keepdims=True)
            xmean = (percent*mask).sum(axis=(1, 2), keepdims=True) / np.maximum(counts, 1)
            ymean = (correct*mask).sum(axis=(1, 2), keepdims=True) / np.maximum(counts, 1)
            xr, yr = (percent-xmean)[mask], (correct-ymean)[mask]
            slope = float(np.dot(xr, yr)/np.dot(xr, xr)*1000) if np.dot(xr, xr) else None
            diagnostics[str(h)][scope] = dict(eligible_rollouts=int(eligible.sum()),
                within_prefix_accuracy_pp_per_10pct_more_shortening=slope,
                within_prefix_all_rollouts=within_prefix_clustered_association(
                    percent, correct, eligible
                ),
                within_prefix_shorter_rollouts=within_prefix_clustered_association(
                    percent, correct, eligible & (delta > 0)
                ))
        print(f'h{h}: validated {length.size:,} rollouts', flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'buckets.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(tables[0]))
        writer.writeheader()
        writer.writerows(tables)
    metadata = dict(baseline_accuracy=float(base_acc.mean()), prefixes=n,
        unique_problems=len(set(problem_ids)), horizons=diagnostics,
        definitions=dict(length='prefix + bridge + student continuation',
            reduction='eight direct rollout mean length minus individual after-bridge length',
            clean='exclude prefix if any direct rollout hits length; exclude individual after rollout if it hits length',
            weighting='individual rollout; baseline accuracy matched to the prefixes represented in each bucket',
            inference='descriptive association only; baseline is estimated from eight samples; no independence assumption or causal claim'),
        sources=[dict(path=str(p), bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns) for p in sources])
    (args.output/'summary.json').write_text(json.dumps(metadata, indent=2))
    if os.environ.get("PRINT_SUMMARY"):
        print(json.dumps(metadata, ensure_ascii=False, separators=(",", ":"), sort_keys=True))
    report = ['# Individual rollout shortening and correctness', '',
        'Baseline: mean full-trace length of eight direct student rollouts for the same prefix.',
        'Each after-bridge rollout is classified separately. Positive reduction means shorter.',
        'Results are descriptive: outcome-conditioned groups do not estimate a causal benefit.', '']
    for scope in ('all', 'clean'):
        for kind in ('overview', 'relative', 'absolute', 'shorter_quintile'):
            report += [f'## {scope}: {kind}', '', '| H | Bucket | N | Correct | Accuracy % | Matched baseline % | Difference pp | Mean reduction % |',
                       '|---:|---|---:|---:|---:|---:|---:|---:|']
            for r in tables:
                if r['scope'] == scope and r['grouping'] == kind:
                    report.append(f"| {r['horizon']} | {r['bucket']} | {r['rollouts']} | {r['correct']} | {r['accuracy_pct']:.3f} | {r['matched_baseline_accuracy_pct']:.3f} | {r['delta_vs_matched_baseline_pp']:.3f} | {r['mean_reduction_pct']:.3f} |")
            report.append('')
    (args.output/'report.md').write_text('\n'.join(report))


if __name__ == '__main__':
    main()
