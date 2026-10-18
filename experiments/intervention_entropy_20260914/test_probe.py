"""CPU-only semantic checks; synthetic fixtures are never experiment results."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location('probe', Path(__file__).with_name('entropy_probe.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class ProbeTest(unittest.TestCase):
    def test_sampling_alignment_and_paired_statistics(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            out = root / 'probe'
            roots = [root/'opd_eval_hard/nojudge_grid_20260912', root/'opd_eval_code/code_reasoning_grid_20260913']
            for ri, r in enumerate(roots):
                datasets = m.DATASETS[:3] if ri == 0 else m.DATASETS[3:]
                r.mkdir(parents=True)
                rows = [dict(dataset=ds, dataset_index=i, problem_id=ds+'-0', prompt_token_ids=[7,8])
                        for i, ds in enumerate(datasets)]
                (r/'prompts.jsonl').write_text('\n'.join(json.dumps(x) for x in rows))
                for t in m.TS:
                    for k in m.KS:
                        config = f'k{k}_t{t:g}'
                        group = r/(config.replace('.', 'p') if ri == 0 else config)
                        m.save(group/'manifest.json', dict(thinking=False, first_token_ban=True,
                               sampling=dict(temperature=1.,top_p=1.), settings=dict(teacher_tokens=k,gap_threshold=t)))
                        for row in rows:
                            trace = dict(row, status='COMPLETE', judge_calls=0, finish_reason='stop',
                                 response_token_ids=[9]+[10]*k+[11],
                                 token_sources=['student']+['teacher']*k+['student'],
                                 events=[dict(anchor_response_position=1,bridge_token_ids=[10]*k)])
                            m.save(group/'completed'/f"{row['dataset_index']:05d}.json",trace)
            args=SimpleNamespace(root=root,out=out,n=1)
            m.prepare(args)
            frozen=(out/'inputs.json').read_bytes()
            m.prepare(args)
            self.assertEqual(frozen,(out/'inputs.json').read_bytes())
            data=json.loads(frozen)
            self.assertEqual(len(data['tasks']),200)
            for task in data['tasks']:
                for point in task['probe_points']:
                    self.assertEqual(point['offset'], 1 if point['phase']=='before' else 1+point['probe_k'])
                for role in ['student','teacher']:
                    pts=[dict(p,entropy_nats=(2. if role=='student' else 1.)) for p in task['probe_points']]
                    m.save(out/role/(task['task_id']+'.json'),dict(manifest_hash=data['manifest_hash'],points=pts,rejected=[]))
            m.summarize(args)
            with (out/'summary.csv').open() as f:
                summary=list(m.csv.DictReader(f))
            self.assertEqual(len(summary),800)
            self.assertTrue(all(float(r['student_higher_after'])==1 for r in summary))
            self.assertTrue(all(float(r['student_delta'])==0 for r in summary))
            with (out/'k_contrasts.csv').open() as f:
                contrasts=list(m.csv.DictReader(f))
            self.assertTrue(all(float(r['student_higher_fraction_difference'])==0 for r in contrasts))

    def test_wilson(self):
        lo,hi=m.wilson(10,20)
        self.assertLess(lo,.5)
        self.assertGreater(hi,.5)
        self.assertEqual(m.wilson(0,0),(None,None))


if __name__=='__main__':
    unittest.main()
