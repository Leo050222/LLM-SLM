"""Read-only reanalysis of saved rollouts; no model/API calls.

Run from any directory. Outputs aggregate JSON beside this script.
Bootstrap resamples paired questions within a benchmark, conditional on the
existing single-seed outputs; it does not estimate generation-seed variation.
"""
import collections
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def read_scores(relative, default=None):
    result = {}
    for path in sorted((ROOT / relative).glob('*.json')):
        row = json.loads(path.read_text())
        # Documented extraction-empty student/Olympiad answer remains incorrect
        # in the full denominator; it is not a missing generation.
        known_empty = (relative == 'experiments/hard_three_20260911/artifacts/scores/student'
                       and path.name == '00309.json' and row['status'] == 'missing'
                       and row['correct'] is False)
        assert row['status'] == 'ok' or known_empty, path
        dataset = row.get('dataset', default)
        problem_id = str(row['problem_id'])
        # The combined suite prepends its dataset name to MATH500's native ID.
        if dataset == 'math500' and not problem_id.startswith('math500/'):
            problem_id = 'math500/' + problem_id
        key = (dataset, problem_id)
        assert key not in result, key
        result[key] = int(row['correct'])
    return result


def main():
    roots = [
        ('nojudge', 'experiments/threshold_k8_nojudge_20260911/artifacts/online/completed'),
        ('judge_hard', 'experiments/hard_three_20260911/artifacts/online/completed'),
        ('judge_math', 'experiments/math500_three_20260911/results/online/completed'),
    ]
    metrics = {}
    for group, relative in roots:
        stats = collections.defaultdict(collections.Counter)
        for path in sorted((ROOT / relative).glob('*.json')):
            row = json.loads(path.read_text())
            s = stats[row.get('dataset', 'math500')]
            assert row['status'] == 'COMPLETE', path
            assert len(row['token_sources']) == len(row['response_token_ids'])
            student_retained = row['token_sources'].count('student')
            assert student_retained + row['discarded_tokens'] == row['student_draft_tokens']
            s['questions'] += 1
            s['output_tokens'] += len(row['response_token_ids'])
            s['teacher_tokens'] += row['token_sources'].count('teacher')
            for field in ['student_draft_tokens', 'discarded_tokens', 'judge_calls', 'rounds']:
                s[field] += row[field]
            for event in row['events']:
                if 'anchor_response_position' not in event:
                    continue
                s['interventions'] += 1
                candidate = event.get('threshold_candidate')
                if candidate is None:
                    selected = [j['candidate'] for j in event.get('judgments', [])
                                if j['judgment'].get('select_anchor')]
                    assert len(selected) == 1, path
                    candidate = selected[0]
                gap = candidate['signed_logprob_gap_student_minus_teacher']
                s['positive_gap' if gap > 0 else 'negative_gap'] += 1
                s['teacher_original_top1'] += int(
                    candidate['teacher_top_tokens'][0]['token_id'] == candidate['student_token_id'])
        for dataset, s in stats.items():
            metrics[group + '/' + dataset] = dict(s,
                teacher_output_fraction=s['teacher_tokens'] / s['output_tokens'],
                discarded_draft_fraction=s['discarded_tokens'] / s['student_draft_tokens'],
                negative_intervention_fraction=s['negative_gap'] / s['interventions'],
                mean_teacher_tokens=s['teacher_tokens'] / s['questions'],
                mean_rounds=s['rounds'] / s['questions'])

    hard = 'experiments/hard_three_20260911/artifacts/scores/'
    math = 'experiments/math500_three_20260911/results/scores/'
    student = read_scores(hard + 'student')
    student.update(read_scores(math + 'student', 'math500'))
    judge = read_scores(hard + 'online')
    judge.update(read_scores(math + 'online', 'math500'))
    nojudge = read_scores('experiments/threshold_k8_nojudge_20260911/artifacts/scores/online')
    teacher = read_scores(hard + 'teacher')
    assert len(nojudge) == len(student) == len(judge) == 2314
    assert nojudge.keys() == student.keys() == judge.keys()
    rng = np.random.default_rng(20260911)
    comparisons = []
    for label, baseline in [('student', student), ('judge', judge), ('teacher', teacher)]:
        datasets = sorted({k[0] for k in nojudge} & {k[0] for k in baseline})
        for dataset in datasets:
            keys = sorted(k for k in nojudge if k[0] == dataset)
            assert all(k in baseline for k in keys)
            left = np.array([nojudge[k] for k in keys])
            right = np.array([baseline[k] for k in keys])
            n = len(keys)
            diff = left - right
            counts = np.bincount(diff + 1, minlength=3)
            # Sufficient-statistic equivalent of paired question bootstrap.
            bootstrap = rng.multinomial(n, counts / n, size=20000)
            deltas = (bootstrap[:, 2] - bootstrap[:, 0]) / n
            comparisons.append(dict(
                comparison='nojudge_minus_' + label, dataset=dataset, n=n,
                rescued=int(sum((left == 1) & (right == 0))),
                harmed=int(sum((left == 0) & (right == 1))),
                delta_pp=float(diff.mean()) * 100,
                question_paired_bootstrap_95ci_pp=(np.quantile(deltas, [.025, .975]) * 100).tolist()))
    result = dict(
        note='Exploratory, paired-question uncertainty conditional on existing single-seed outputs. '
             'No multiplicity correction, no inference of causal location effects or equal-budget superiority. '
             'BBH uses the existing fixed 27-task mixture; no unseen-task generalization inference.',
        metrics=metrics, paired_comparisons=comparisons)
    (OUT / 'reanalysis.json').write_text(json.dumps(result, indent=2) + '\n')
    print('Verified and aggregated 4628 completed rollouts; aligned 2314 question IDs across three policies.')


if __name__ == '__main__':
    main()
