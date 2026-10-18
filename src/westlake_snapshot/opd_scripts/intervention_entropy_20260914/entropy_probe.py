"""Read-only, full-vocabulary entropy probes on frozen intervention rollouts."""
import argparse
import collections
import csv
import hashlib
import json
import math
import random
import time
from pathlib import Path

KS = [1, 4, 8, 16, 32]
TS = [2, 2.5, 3, 3.5, 4]
DATASETS = ['math500', 'olympiad_bench', 'bbh', 'cruxeval_o', 'cruxeval_i',
            'humaneval_plus', 'mbpp_plus', 'livecodebench']
SEED = 20260914


def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()


def save(path, x):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(x, ensure_ascii=False))
    tmp.replace(path)


def prepare(args):
    roots = [args.root / 'opd_eval_hard/nojudge_grid_20260912',
             args.root / 'opd_eval_code/code_reasoning_grid_20260913']
    tasks, cohorts, selection, source_manifests = [], [], {}, {}
    for root in roots:
        prompts = [json.loads(s) for s in (root / 'prompts.jsonl').read_text().splitlines() if s]
        selected = []
        for ds in DATASETS:
            pool = [r for r in prompts if r['dataset'] == ds]
            if not pool:
                continue
            chosen = sorted(pool, key=lambda r: digest([SEED, ds, r['problem_id']]))[:args.n]
            assert len(chosen) == args.n
            selection[ds] = [r['problem_id'] for r in chosen]
            selected += chosen
        for t in TS:
            for k in KS:
                config = f'k{k}_t{t:g}'
                source_config = config.replace('.', 'p') if root == roots[0] else config
                group = root / source_config
                manifest = json.loads((group / 'manifest.json').read_text())
                assert manifest['thinking'] is False and manifest['first_token_ban'] is True
                assert manifest['sampling'] == {'temperature': 1., 'top_p': 1.}
                assert manifest['settings']['teacher_tokens'] == k
                assert manifest['settings']['gap_threshold'] == t
                source_manifests[str(group)] = digest(manifest)
                for row in selected:
                    path = group / 'completed' / f"{row['dataset_index']:05d}.json"
                    blob = path.read_bytes()
                    trace = json.loads(blob)
                    for key in ['problem_id', 'dataset', 'prompt_token_ids']:
                        assert trace[key] == row[key], (path, key)
                    assert trace['status'] == 'COMPLETE' and trace['judge_calls'] == 0
                    events = [e for e in trace['events'] if 'bridge_token_ids' in e]
                    cohort = dict(dataset=row['dataset'], problem_id=row['problem_id'],
                                  config=config, threshold=t, k=k, events=len(events),
                                  finish_reason=trace.get('finish_reason'), points=0)
                    cohorts.append(cohort)
                    if not events:
                        continue
                    response = trace['response_token_ids']
                    sources = trace['token_sources']
                    rng = random.Random(digest([SEED, config, row['dataset'], row['problem_id'], 'event']))
                    picked = [('first', 0), ('uniform', rng.randrange(len(events)))]
                    points = []
                    for sampling, ei in picked:
                        event = events[ei]
                        anchor = event['anchor_response_position']
                        bridge = event['bridge_token_ids']
                        actual = len(bridge)
                        assert response[anchor:anchor+actual] == bridge
                        assert sources[anchor:anchor+actual] == ['teacher'] * actual
                        assert 0 < actual <= k
                        common = dict(selection=sampling, event_index=ei, anchor=anchor,
                                      actual_k=actual, requested_k=k)
                        # EOS is marked at score time using the frozen model generation config.
                        points.append(dict(common, view='actual', probe_k=k, offset=anchor,
                                           phase='before', terminal_candidate=False))
                        points.append(dict(common, view='actual', probe_k=k, offset=anchor+actual,
                                           phase='after', terminal_candidate=True))
                        if k == 32 and actual == 32:
                            for probe_k in KS:
                                points.append(dict(common, view='matched32', probe_k=probe_k,
                                                   offset=anchor, phase='before', terminal_candidate=False))
                                points.append(dict(common, view='matched32', probe_k=probe_k,
                                                   offset=anchor+probe_k, phase='after',
                                                   terminal_candidate=True))
                    cohort['points'] = len(points)
                    task = dict(cohort, source=str(path), source_sha256=hashlib.sha256(blob).hexdigest(),
                                prompt_token_ids=row['prompt_token_ids'], response_token_ids=response,
                                probe_points=points)
                    task['task_id'] = digest([config, row['dataset'], row['problem_id']])[:24]
                    tasks.append(task)
    assert len(cohorts) == args.n * 8 * 25
    payload = dict(seed=SEED, n_per_dataset=args.n, selection=selection,
                   source_manifests=source_manifests, cohorts=cohorts, tasks=tasks)
    payload['manifest_hash'] = digest(payload)
    target = args.out / 'inputs.json'
    if target.exists():
        assert json.loads(target.read_text()) == payload, 'Frozen sample changed'
    else:
        save(target, payload)
    print(json.dumps(dict(traces=len(cohorts), tasks=len(tasks), points=sum(c['points'] for c in cohorts))), flush=True)


def score(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    data = json.loads((args.out / 'inputs.json').read_text())
    model_path = '/storage/yukaichengLab/share/model/Qwen3-' + ('1.7B' if args.role == 'student' else '14B')
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    other = AutoTokenizer.from_pretrained('/storage/yukaichengLab/share/model/Qwen3-1.7B', local_files_only=True)
    assert tok.get_vocab() == other.get_vocab(), 'Vocabulary alignment mismatch'
    torch.manual_seed(SEED)
    x = torch.randn(256, 256, device='cuda')
    assert torch.isfinite(x @ x).all()
    print('CUDA_WITNESS_OK', torch.cuda.get_device_name(), flush=True)
    del x
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
             dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda').eval()
    model.requires_grad_(False)
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    started = time.monotonic()
    save(args.out / f'{args.role}_runtime.json', dict(model=model_path, torch=torch.__version__,
         vocab_sha256=digest(tok.get_vocab()), vocab_size=model.config.vocab_size,
         eos=sorted(eos), manifest_hash=data['manifest_hash'], device=torch.cuda.get_device_name()))
    with torch.inference_mode():
        for i, task in enumerate(data['tasks']):
            target = args.out / args.role / (task['task_id'] + '.json')
            if target.exists():
                old = json.loads(target.read_text())
                assert old['manifest_hash'] == data['manifest_hash']
                continue
            prompt = task['prompt_token_ids']
            response = task['response_token_ids']
            valid, rejected = [], []
            for point in task['probe_points']:
                p = dict(point)
                end = point['offset']
                if point['phase'] == 'after' and point['view'] == 'actual' and point['actual_k'] != point['probe_k']:
                    p['reason'] = 'bridge_shorter_than_requested_k'
                    rejected.append(p)
                elif point['terminal_candidate'] and response[end-1] in eos:
                    p['reason'] = 'teacher_ended_with_eos'
                    rejected.append(p)
                else:
                    valid.append(p)
            # Fair matched32 analysis requires every compared horizon to be valid.
            bad = {p['selection'] for p in rejected if p['view'] == 'matched32'}
            retained = []
            for p in valid:
                if p['view'] == 'matched32' and p['selection'] in bad:
                    rejected.append(dict(p, reason='matched32_requires_all_five_horizons'))
                else:
                    retained.append(p)
            valid = retained
            positions = sorted({len(prompt) + p['offset'] - 1 for p in valid})
            assert positions and positions[0] >= 0
            ids = (prompt + response)[:positions[-1]+1]
            assert len(ids) <= 40960
            input_ids = torch.tensor([ids], device='cuda', dtype=torch.long)
            t0 = time.monotonic()
            hidden = model.model(input_ids=input_ids, use_cache=False, return_dict=True).last_hidden_state[0]
            selected = hidden[positions]
            del hidden
            logits = model.lm_head(selected).float()
            lp = torch.log_softmax(logits, dim=-1)
            entropy = -(lp.exp() * lp).sum(-1)
            assert torch.isfinite(entropy).all()
            assert (entropy >= 0).all() and (entropy <= math.log(logits.shape[-1])+1e-4).all()
            values = dict(zip(positions, entropy.tolist()))
            if i == 0:
                # Check indexing/causal prefix independence with a separate truncated forward.
                pos = positions[0]
                h = model.model(input_ids=input_ids[:, :pos+1], use_cache=False,
                                return_dict=True).last_hidden_state[:, -1]
                check_lp = torch.log_softmax(model.lm_head(h).float(), dim=-1)
                check = -(check_lp.exp()*check_lp).sum().item()
                delta = abs(check-values[pos])
                assert delta < .05, ('causal_alignment', delta)
                save(args.out / f'{args.role}_alignment_witness.json', dict(delta_nats=delta, passed=True))
                del h, check_lp
            for p in valid:
                p['entropy_nats'] = values[len(prompt)+p['offset']-1]
            save(target, dict(manifest_hash=data['manifest_hash'], task_id=task['task_id'],
                             role=args.role, points=valid, rejected=rejected,
                             seconds=time.monotonic()-t0, prefix_tokens=len(ids)))
            del selected, logits, lp, entropy, input_ids
            if (i+1) % 20 == 0 or i == 0:
                print(json.dumps(dict(role=args.role, completed=i+1, total=len(data['tasks']),
                                      seconds=time.monotonic()-started)), flush=True)
    save(args.out / f'{args.role}_COMPLETE.json', dict(tasks=len(data['tasks']), manifest_hash=data['manifest_hash']))


def wilson(success, n):
    if not n:
        return None, None
    p, z = success/n, 1.96
    d = 1+z*z/n
    middle = (p+z*z/(2*n))/d
    radius = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return middle-radius, middle+radius


def summarize(args):
    data = json.loads((args.out / 'inputs.json').read_text())
    groups, paired_rows = collections.defaultdict(list), []
    for task in data['tasks']:
        s, t = [json.loads((args.out / role / (task['task_id']+'.json')).read_text())
                for role in ['student', 'teacher']]
        assert s['manifest_hash'] == t['manifest_hash'] == data['manifest_hash']
        assert s['rejected'] == t['rejected']
        pairs = collections.defaultdict(dict)
        for sp, tp in zip(s['points'], t['points'], strict=True):
            assert {k:v for k,v in sp.items() if k != 'entropy_nats'} == {k:v for k,v in tp.items() if k != 'entropy_nats'}
            key = (sp['view'], sp['selection'], sp['probe_k'])
            pairs[key][sp['phase']] = (sp['entropy_nats'], tp['entropy_nats'])
        for (view, selection, k), phases in pairs.items():
            if 'after' not in phases:
                continue
            bs, bt = phases['before']; hs, ht = phases['after']
            row = dict(dataset=task['dataset'], threshold=task['threshold'], k=k,
                       view=view, selection=selection, problem_id=task['problem_id'],
                       student_before=bs, teacher_before=bt, student_after=hs, teacher_after=ht,
                       student_delta=hs-bs, teacher_delta=ht-bt, entropy_gap_after=hs-ht,
                       student_higher_before=int(bs>bt), student_higher_after=int(hs>ht),
                       source=task['source'])
            groups[task['dataset'], task['threshold'], k, view, selection].append(row)
            paired_rows.append(row)
    summary = []
    metrics = ['student_before','teacher_before','student_after','teacher_after',
               'student_delta','teacher_delta','entropy_gap_after','student_higher_before','student_higher_after']
    for key, rows in sorted(groups.items()):
        n = len(rows)
        assert len({r['problem_id'] for r in rows}) == n
        lo, hi = wilson(sum(r['student_higher_after'] for r in rows), n)
        summary.append(dict(zip(['dataset','threshold','k','view','selection'], key), n=n,
                            **{m:sum(r[m] for r in rows)/n for m in metrics},
                            student_higher_after_ci95_low=lo, student_higher_after_ci95_high=hi))
    # Pairwise k contrasts use only shared questions, for each threshold/dataset/view.
    contrasts=[]
    for ds in DATASETS:
        for t in TS:
            for view in ['actual','matched32']:
                for selection in ['first','uniform']:
                    base={r['problem_id']:r for r in groups.get((ds,t,1,view,selection), [])}
                    for k in KS[1:]:
                        other={r['problem_id']:r for r in groups.get((ds,t,k,view,selection), [])}
                        ids=sorted(base.keys() & other.keys())
                        if not ids:
                            continue
                        differences=[other[q]['student_higher_after']-base[q]['student_higher_after'] for q in ids]
                        rng=random.Random(digest([SEED,ds,t,k,view,selection,'bootstrap']))
                        boot=sorted(sum(rng.choices(differences,k=len(ids)))/len(ids) for _ in range(2000))
                        contrasts.append(dict(dataset=ds,threshold=t,k=k,reference_k=1,view=view,
                             selection=selection,n_shared=len(ids),
                             student_higher_fraction_difference=sum(differences)/len(ids),
                             bootstrap_ci95_low=boot[49],bootstrap_ci95_high=boot[1949]))
    for name, rows in [('summary',summary),('paired_points',paired_rows),('cohorts',data['cohorts']),('k_contrasts',contrasts)]:
        with (args.out/(name+'.csv')).open('w', newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    save(args.out/'COMPLETE.json',dict(tasks=len(data['tasks']),summary_rows=len(summary),
         manifest_hash=data['manifest_hash'], units='nats', full_vocabulary=True))
    print('COMPLETE',len(summary),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['prepare','score','summarize'])
    p.add_argument('--root',type=Path,default=Path('/storage/yukaichengLab/wangxin/test_eiloeo'))
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--n',type=int,default=20)
    p.add_argument('--role',choices=['student','teacher'])
    args=p.parse_args()
    globals()[args.action](args)
