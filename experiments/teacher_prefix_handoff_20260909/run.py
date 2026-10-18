"""One student completion per frozen teacher prefix; resume atomic conditions."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path('/storage/yukaichengLab/wangxin/test_eiloeo')
BASE = ROOT / 'opd_eval_deepmath/ge6_test1000_native_nonthinking_seed20260811'
MODEL = '/storage/yukaichengLab/share/model/Qwen3-1.7B'
LENGTHS = (32, 64, 128, 256)

def read(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def atomic(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    tmp.replace(path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--limit', type=int, default=1000)
    args = ap.parse_args()
    assert 1 <= args.limit <= 1000
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from run_opd_logit_diagnostic import score_math_response, deepmath_user_prompt
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    teachers = read(BASE / 'qwen3_14b/merged/rollouts.jsonl')
    students = read(BASE / 'qwen3_1_7b/merged/rollouts.jsonl')
    teachers.sort(key=lambda r: int(r['dataset_index']))
    students.sort(key=lambda r: int(r['dataset_index']))
    assert [int(r['dataset_index']) for r in teachers] == list(range(1000))
    assert len(students) == 1000
    assert sum(bool(r['correct']) for r in teachers) == 436
    assert sum(bool(r['correct']) for r in students) == 283
    for t, s in zip(teachers, students):
        for k in ('dataset_index', 'problem_id', 'problem', 'solution', 'prompt_token_ids'):
            assert t[k] == s[k], (k, t['dataset_index'])
        encoded = tok.apply_chat_template([{'role':'user', 'content':deepmath_user_prompt(t['problem'])}], tokenize=True, add_generation_prompt=True, enable_thinking=False)
        ids = encoded['input_ids'] if hasattr(encoded, 'keys') else encoded
        assert list(ids) == t['prompt_token_ids']
        assert len(t['response_token_ids']) == t['response_tokens']
        assert t['finish_reason'] == 'stop'
    args.output.mkdir(parents=True, exist_ok=True)
    config = {'teacher_sha256':digest(BASE/'qwen3_14b/merged/rollouts.jsonl'), 'student_sha256':digest(BASE/'qwen3_1_7b/merged/rollouts.jsonl'), 'model':MODEL, 'lengths':LENGTHS, 'limit':args.limit, 'seed':20260811, 'temperature':1.0, 'top_p':1.0, 'top_k':-1, 'max_model_len':40960, 'total_response_budget':40264, 'n':1, 'script_sha256':digest(Path(__file__)), 'scorer_sha256':digest(ROOT/'opd_scripts/run_opd_logit_diagnostic.py'), 'short_teacher_policy':'retain finished teacher answer; no continuation'}
    config = json.loads(json.dumps(config))
    cp = args.output/'config.json'
    if cp.exists():
        assert json.loads(cp.read_text()) == config
    else:
        atomic(cp, config)
    teachers, students = teachers[:args.limit], students[:args.limit]
    engine = None
    results = []
    for h in LENGTHS:
        dest = args.output/f'h{h}.json'
        if dest.exists():
            rows = json.loads(dest.read_text())
        else:
            rows, pending, prompts, params = [], [], [], []
            for t in teachers:
                prefix = t['response_token_ids'][:h]
                finished = len(t['response_token_ids']) <= h or tok.eos_token_id in prefix
                row = {'dataset_index':t['dataset_index'], 'problem_id':t['problem_id'], 'prefix_length_requested':h, 'prefix_token_ids':prefix, 'actual_prefix_length':len(prefix), 'teacher_finished_before_handoff':finished, 'solution':t['solution'], 'teacher_correct':bool(t['correct'])}
                if finished:
                    row.update(continuation_token_ids=[], response=t['response'], finish_reason='teacher_stop', correct=bool(t['correct']), score_method='frozen_teacher_score')
                    rows.append(row)
                else:
                    ids = t['prompt_token_ids'] + prefix
                    budget = min(40264-len(prefix), 40960-len(ids))
                    assert budget > 0
                    pending.append(row)
                    prompts.append(TokensPrompt(prompt_token_ids=ids))
                    params.append(SamplingParams(temperature=1.0, top_p=1.0, top_k=-1, max_tokens=budget, n=1, seed=20260811))
            if prompts:
                if engine is None:
                    import torch
                    torch.manual_seed(20260811)
                    assert int(torch.arange(8,device='cuda').sum()) == 28
                    print('CUDA_WITNESS_OK',torch.cuda.get_device_name(),flush=True)
                    engine = LLM(model=MODEL, tokenizer=MODEL, dtype='bfloat16', tensor_parallel_size=1, seed=20260811, max_model_len=40960, max_num_seqs=8, gpu_memory_utilization=.90, enable_prefix_caching=True)
                outputs = engine.generate(prompts, params)
                for row, out in zip(pending, outputs, strict=True):
                    gen = out.outputs[0]
                    cont = list(gen.token_ids)
                    response = tok.decode(row['prefix_token_ids']+cont, skip_special_tokens=False)
                    row.update(continuation_token_ids=cont, response=response, finish_reason=gen.finish_reason)
                    rows.append(row)
            rows.sort(key=lambda r:int(r['dataset_index']))
            atomic(args.output/f'h{h}.unscored.json', rows)
            for row in rows:
                if 'correct' not in row:
                    correct, predicted, method = score_math_response(row['response'],row['solution'],20)
                    row.update(correct=bool(correct), predicted_answer=predicted, score_method=method)
            atomic(dest,rows)
        assert [int(r['dataset_index']) for r in rows] == list(range(args.limit))
        assert all(r['prefix_token_ids'] == t['response_token_ids'][:h] for r,t in zip(rows,teachers))
        correct = sum(r['correct'] for r in rows)
        rescued = sum(r['correct'] and not s['correct'] for r,s in zip(rows,students))
        harmed = sum(not r['correct'] and s['correct'] for r,s in zip(rows,students))
        results.append({'h':h,'n':len(rows),'correct':correct,'accuracy':correct/len(rows),'delta_pp_vs_student':100*(correct-sum(s['correct'] for s in students))/len(rows),'rescued':rescued,'harmed':harmed,'teacher_already_finished':sum(r['teacher_finished_before_handoff'] for r in rows),'length_stops':sum(r['finish_reason']=='length' for r in rows),'score_timeouts_or_errors':sum(r['score_method'] in ('timeout','error') for r in rows)})
        atomic(args.output/'summary.json',results)
        print(results[-1],flush=True)
    (args.output/'COMPLETE').write_text('verified all four conditions\n')

if __name__ == '__main__':
    main()
