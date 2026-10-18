#!/usr/bin/env python3
"""Persistent vLLM worker. Launch one student and one teacher on separate GPUs."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, '/storage/yukaichengLab/wangxin/test_eiloeo/opd_online_code/20260911_chunk64_k8_2nodes/ocrd_pipeline')
from online_rollout.core import atomic_json, digest
from anchor_selector.core import lookup_actual_logprob, sorted_topk, tokenizer_fingerprint


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--queue', type=Path, required=True)
    p.add_argument('--role', choices=['student', 'teacher'], required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--max-model-len', type=int, default=40960)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--gpu-memory-utilization', type=float, default=.9)
    args = p.parse_args()
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    import torch
    torch.manual_seed(20260806)
    x = torch.randn(256, 256, device='cuda')
    y = x @ x
    torch.cuda.synchronize()
    print('CUDA_WITNESS_OK', torch.cuda.get_device_name(), float(y.sum()), flush=True)
    del x, y
    import fcntl
    base = args.queue / args.role
    (base / 'requests').mkdir(parents=True, exist_ok=True)
    (base / 'results').mkdir(exist_ok=True)
    lock = (base / 'worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    engine = LLM(model=args.model, tensor_parallel_size=1, dtype='bfloat16',
                 max_model_len=args.max_model_len, max_num_seqs=args.batch_size,
                 gpu_memory_utilization=args.gpu_memory_utilization,
                 enable_prefix_caching=False, max_logprobs=16, disable_log_stats=True)
    tokenizer = engine.get_tokenizer()
    vocab_size = len(tokenizer)
    # Disable prefix caching initially: scoring requires aligned full prompt logprobs.
    info = dict(model=args.model, tokenizer_sha256=tokenizer_fingerprint(tokenizer),
                max_model_len=args.max_model_len, protocol=1,
                worker_sha256=digest(Path(__file__).read_text()),
                vllm_version=vllm.__version__,
                eos_token_id=tokenizer.eos_token_id)
    info_path = base / 'model.json'
    if info_path.exists() and json.loads(info_path.read_text()) != info:
        raise ValueError('Queue belongs to a different model/config; use a new queue')
    atomic_json(info_path, info)
    atomic_json(base / 'ready.json', info)
    while not (args.queue / 'STOP').exists():
        paths = sorted((base / 'requests').glob('*.json'))[:args.batch_size]
        if not paths:
            time.sleep(.1)
            continue
        pending, prompts, params = [], [], []
        for path in paths:
            target = base / 'results' / path.name
            if target.exists():
                path.unlink(missing_ok=True)
                continue
            try:
                req = json.loads(path.read_text())
                if req['model_identity'] != digest(info):
                    raise ValueError('Model identity mismatch')
                ids = req['prefix']
                if req['op'] == 'score':
                    ids = ids + req['tokens']
                    sampling = SamplingParams(max_tokens=1, temperature=1., top_p=1.,
                                              prompt_logprobs=16, seed=0)
                elif req['op'] == 'generate':
                    extra = {}
                    if req.get('banned_first') is not None:
                        # Request is one token long: the ban applies only at bridge entry.
                        if req['n'] != 1:
                            raise ValueError('First-token ban requires n=1')
                        extra['_bad_words_token_ids'] = [[req['banned_first']]]
                    sampling = SamplingParams(max_tokens=req['n'], temperature=1., top_p=1.,
                                              seed=req['seed'], logprobs=1, **extra)
                else:
                    raise ValueError('Unknown operation')
                if not ids or len(ids) + sampling.max_tokens > args.max_model_len:
                    raise ValueError('Request exceeds model context')
                pending.append((path, req, ids))
                prompts.append(TokensPrompt(prompt_token_ids=ids))
                params.append(sampling)
            except Exception as error:
                atomic_json(target, {'error': f'{type(error).__name__}: {error}'})
                path.unlink(missing_ok=True)
        if not pending:
            continue
        try:
            batch_started = time.monotonic()
            outputs = engine.generate(prompts, params, use_tqdm=False)
            batch_seconds = time.monotonic() - batch_started
            if len(outputs) != len(pending):
                raise ValueError('Worker output count mismatch')
            for (path, req, ids), output in zip(pending, outputs, strict=True):
                if req['op'] == 'score':
                    entries = output.prompt_logprobs
                    if entries is None or len(entries) != len(ids):
                        raise ValueError('Misaligned prompt logprobs')
                    chosen, top_ids, top_lp = [], [], []
                    for pos in range(len(req['prefix']), len(ids)):
                        chosen.append(lookup_actual_logprob(entries[pos], ids[pos]))
                        # Student top-k is never consumed by Backend.candidates.
                        # Keep the actual-token logprob exact; teacher top16 stays strict.
                        ii, ll = ([], []) if args.role == 'student' else sorted_topk(entries[pos], 16, vocab_size)
                        top_ids.append(ii)
                        top_lp.append(ll)
                    value = dict(chosen=chosen, top_ids=top_ids, top_lp=top_lp)
                else:
                    generated = output.outputs[0]
                    value = dict(token_ids=list(generated.token_ids),
                                 finish_reason=generated.finish_reason)
                value['worker_batch_seconds'] = batch_seconds
                value['worker_batch_size'] = len(pending)
                atomic_json(base / 'results' / path.name, value)
                path.unlink(missing_ok=True)
        except Exception as error:
            for path, _, _ in pending:
                target = base / 'results' / path.name
                if not target.exists():
                    atomic_json(target, {'error': f'{type(error).__name__}: {error}'})
                path.unlink(missing_ok=True)
            raise


if __name__ == '__main__':
    main()
