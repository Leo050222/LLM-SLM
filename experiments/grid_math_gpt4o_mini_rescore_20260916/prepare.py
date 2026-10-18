import collections, csv, hashlib, json, os, sys
from pathlib import Path
from types import SimpleNamespace

ROOT=Path('/storage/yukaichengLab/wangxin/test_eiloeo')
ARCHIVE=ROOT/'rollout_archive/grid_20260912_20260913'
OUT=ROOT/'opd_rescore/grid_math_gpt4o_mini_openlux_20260916'
sys.path.insert(0,str(ROOT/'eval_repos/evalscope'))
from evalscope.metrics.math.parser import extract_answer
from evalscope.benchmarks.bbh.bbh_adapter import BBHAdapter
from evalscope.benchmarks.olympiad_bench.olympiad_bench_adapter import OlympiadBenchAdapter

def sha(b):return hashlib.sha256(b).hexdigest()

def main():
    if (OUT/'INPUTS_COMPLETE.json').exists():
        m=json.loads((OUT/'INPUTS_COMPLETE.json').read_text())
        assert sha((OUT/'inputs.jsonl').read_bytes())==m['inputs_sha256']
        print('INPUTS_ALREADY_VERIFIED',m['rows'],flush=True);return
    counts=collections.Counter();methods=collections.Counter();failures=[];n=0
    temp=OUT/'inputs.jsonl.tmp'
    with (ARCHIVE/'index.csv').open() as index,temp.open('w') as dest:
        for r in csv.DictReader(index):
            if r['category']!='math':continue
            source=ARCHIVE/r['relative_path'];blob=source.read_bytes();assert sha(blob)==r['sha256']
            x=json.loads(blob);ds=x['dataset'];candidate=None;extraction_status='ok'
            score_path=ROOT/'opd_eval_hard/nojudge_grid_20260912/scores'/r['source_config']/source.name
            old=json.loads(score_path.read_text()) if score_path.exists() else None
            try:
                if old and old.get('source_sha256')==r['sha256'] and old.get('status') in ['ok','missing'] and 'predicted' in old:
                    candidate=old['predicted'];method='cached_old_predicted'
                else:
                    response=x['response'].rsplit('</think>',1)[-1].strip()
                    if ds=='math500':candidate=extract_answer(response)
                    elif ds=='bbh':candidate=(BBHAdapter._extract_mc_answer if x['metadata']['multiple_choice'] else BBHAdapter._extract_ff_answer)(response)
                    elif ds=='olympiad_bench':candidate=OlympiadBenchAdapter.extract_answer(None,response,SimpleNamespace(metadata=x['metadata']))
                    else:raise ValueError('Unknown dataset')
                    method='old_extractor_replayed'
                assert candidate is None or isinstance(candidate,str)
            except Exception as e:
                extraction_status=type(e).__name__;method='extraction_error';failures.append(r['relative_path'])
            reference=x['metadata']['final_answer'][0] if ds=='olympiad_bench' else x['solution']
            assert isinstance(reference,str) and reference.strip()
            payload=dict(candidate_answer=candidate,reference_answer=reference)
            text=json.dumps(payload,ensure_ascii=False,separators=(',',':'))
            row=dict(dataset=ds,config=r['config'],source_config=r['source_config'],problem_id=x['problem_id'],
                     dataset_index=x['dataset_index'],relative_path=r['relative_path'],source_sha256=r['sha256'],
                     finish_reason=x.get('finish_reason'),extraction_method=method,extraction_status=extraction_status,
                     candidate_answer=candidate,reference_answer=reference,user_message=text,input_sha256=sha(text.encode()))
            dest.write(json.dumps(row,ensure_ascii=False)+'\n');n+=1;counts[ds]+=1;methods[method]+=1
            if n%5000==0:print('PREPARED',n,flush=True)
    assert counts=={'math500':12500,'olympiad_bench':16850,'bbh':27000} and n==56350
    assert not failures, f'Extraction errors: {len(failures)}; no API scoring started'
    os.replace(temp,OUT/'inputs.jsonl')
    manifest=dict(rows=n,counts=dict(counts),extraction_methods=dict(methods),inputs_sha256=sha((OUT/'inputs.jsonl').read_bytes()))
    (OUT/'INPUTS_COMPLETE.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest),flush=True)

if __name__=='__main__':main()
