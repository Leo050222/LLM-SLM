"""Freeze the 4,631-row code+math+AIME input manifest for reverse intervention."""
import json, hashlib
from pathlib import Path
from collections import Counter

def main():
    here=Path(__file__).resolve().parent
    rows=[json.loads(line) for line in (here/'prompts.jsonl').read_text().splitlines() if line.strip()]
    assert len(rows)==4631 and len({r['dataset_index'] for r in rows})==4631
    counts=Counter(r['dataset'] for r in rows)
    assert counts['aime24']==30 and counts['aime25']==30
    raw=(here/'prompts.jsonl').read_bytes()
    meta={'rows':len(rows),'sha256':hashlib.sha256(raw).hexdigest(),'counts':dict(counts),
          'main_model':'Qwen3-1.7B','intervention_model':'Qwen3-8B','threshold':2.0,'k':1,'topk':100,
          'sampling':{'temperature':0.0,'top_p':1.0,'top_k':-1},
          'trigger':'main_model_logprob(main_token) - intervention_model_logprob(main_token) > 2',
          'intervention':'main model prompt repair; original main token banned; exactly one replacement token'}
    (here/'INPUTS_COMPLETE.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(meta,ensure_ascii=False))
if __name__=='__main__':main()
