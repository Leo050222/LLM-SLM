#!/usr/bin/env python3
"""Persistent vLLM worker. Launch one student and one teacher on separate GPUs."""
import argparse
import json
from pathlib import Path
import sys
import time
from queue_rpc import claim_batch, publish

sys.path.insert(0, '/storage/yukaichengLab/wangxin/test_eiloeo/opd_online_code/20260911_chunk64_k8_2nodes/ocrd_pipeline')
from online_rollout.core import atomic_json, digest
from anchor_selector.core import lookup_actual_logprob, sorted_topk, tokenizer_fingerprint


def entropy_witness(engine, vocab, base):
    """Compare GPU scalar against the previous full-return CPU calculation."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    import numpy as np
    ids = engine.get_tokenizer().encode('Compute 13 + 29. The answer is')
    prompts = [TokensPrompt(prompt_token_ids=ids), TokensPrompt(prompt_token_ids=ids + ids)]
    engine.collective_rpc('begin_entropy_capture', kwargs={'validate': True})
    try:
        out = engine.generate(prompts, SamplingParams(max_tokens=1, temperature=1., top_p=1.,
            logprobs=vocab, seed=0, detokenize=False), use_tqdm=False)
    finally:
        captured = engine.collective_rpc('end_entropy_capture')[0]
    witnesses = []
    for o, prompt in zip(out, prompts, strict=True):
        entries = o.outputs[0].logprobs[0]
        lp = np.array([v.logprob for v in entries.values()], dtype=np.float64)
        assert len(lp) == vocab
        finite = np.isfinite(lp)
        ref = float(-(lp[finite] * np.exp(lp[finite])).sum())
        actual = captured[o.request_id]
        assert actual['position'] == len(prompt['prompt_token_ids']) - 1
        assert abs(ref - actual['entropy']) < 1e-7, (ref, actual)
        witnesses.append(dict(reference_entropy=ref, **actual))
    # Compact-only pass also checks output request mapping with mixed prefix lengths.
    engine.collective_rpc('begin_entropy_capture', kwargs={'validate': True})
    try:
        compact = engine.generate(prompts, SamplingParams(max_tokens=1, temperature=1., top_p=1.,
            logprobs=1, seed=0, detokenize=False), use_tqdm=False)
    finally:
        values = engine.collective_rpc('end_entropy_capture')[0]
    for o, ref in zip(compact, witnesses, strict=True):
        ref['compact_entropy'] = values[o.request_id]['entropy']
        ref['compact_same_logits_cpu_reference'] = values[o.request_id]['same_logits_cpu_reference']
        ref['independent_forward_delta'] = values[o.request_id]['entropy'] - ref['entropy']
        assert values[o.request_id]['position'] == ref['position']
    # Identical compact-vs-compact forward control documents engine variability.
    engine.collective_rpc('begin_entropy_capture', kwargs={'validate': True})
    try:
        repeated = engine.generate(prompts, SamplingParams(max_tokens=1, temperature=1., top_p=1.,
            logprobs=1, seed=0, detokenize=False), use_tqdm=False)
    finally:
        controls = engine.collective_rpc('end_entropy_capture')[0]
    for o, ref in zip(repeated, witnesses, strict=True):
        assert controls[o.request_id]['position'] == ref['position']
        ref['identical_compact_forward_delta'] = controls[o.request_id]['entropy'] - ref['compact_entropy']
    atomic_json(base / 'ENTROPY_NUMERICAL_WITNESS.json', dict(entropy=witnesses, score_policy='unchanged_top16'))
    print('ENTROPY_NUMERICAL_WITNESS_OK', flush=True)


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
    # vLLM's -1 sampler fast path returns raw tensors without token/rank arrays;
    # the offline RequestOutput converter therefore yields an empty list.
    # Request the exact positive model vocabulary width to use the indexed path.
    full_vocab_size = int(json.loads((Path(args.model) / 'config.json').read_text())['vocab_size'])
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
                 enable_prefix_caching=False, max_logprobs=full_vocab_size, disable_log_stats=True,
                 worker_extension_cls='entropy_extension.EntropyWorkerExtension')
    tokenizer = engine.get_tokenizer()
    vocab_size = len(tokenizer)
    # Disable prefix caching initially: scoring requires aligned full prompt logprobs.
    info = dict(model=args.model, tokenizer_sha256=tokenizer_fingerprint(tokenizer),
                max_model_len=args.max_model_len, protocol=2,
                worker_sha256=digest(Path(__file__).read_text()),
                vllm_version=vllm.__version__,
                eos_token_id=tokenizer.eos_token_id)
    info_path = base / 'model.json'
    if info_path.exists() and json.loads(info_path.read_text()) != info:
        raise ValueError('Queue belongs to a different model/config; use a new queue')
    atomic_json(info_path, info)
    # Readiness is published only after the numerical witness below passes.
    entropy_witness(engine, full_vocab_size, base)
    atomic_json(base / 'ready.json', info)
    while not (args.queue / 'STOP').exists():
        paths = claim_batch(base, args.batch_size)
        if not paths:
            time.sleep(.1)
            continue
        pending, prompts, params = [], [], []
        for path, req, queue_wait in paths:
            target = base / 'results' / path.name
            try:
                if req['model_identity'] != digest(info):
                    raise ValueError('Model identity mismatch')
                ids = req['prefix']
                if req['op'] == 'score':
                    ids = ids + req['tokens']
                    sampling = SamplingParams(max_tokens=1, temperature=1., top_p=1.,
                                              prompt_logprobs=16, seed=0)
                elif req['op'] == 'entropy':
                    sampling = SamplingParams(max_tokens=1, temperature=1., top_p=1.,
                                              logprobs=1, seed=0, detokenize=False)
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
                pending.append((path, req, ids, queue_wait))
                prompts.append(TokensPrompt(prompt_token_ids=ids))
                params.append(sampling)
            except Exception as error:
                publish(base, path, {'error': f'{type(error).__name__}: {error}'})
        if not pending:
            continue
        try:
            batch_started = time.monotonic()
            capture = pending[0][1]['op'] == 'entropy'
            if capture:
                assert all(req['op'] == 'entropy' for _, req, _, _ in pending)
                engine.collective_rpc('begin_entropy_capture')
            try:
                outputs = engine.generate(prompts, params, use_tqdm=False)
            finally:
                if capture:
                    captured = engine.collective_rpc('end_entropy_capture')
                    assert len(captured) == 1
                    captured = captured[0]
            batch_seconds = time.monotonic() - batch_started
            if len(outputs) != len(pending):
                raise ValueError('Worker output count mismatch')
            for (path, req, ids, queue_wait), output in zip(pending, outputs, strict=True):
                post_started = time.monotonic()
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
                elif req['op'] == 'entropy':
                    value = captured[output.request_id]
                    if value['position'] != len(ids) - 1 or value['vocabulary_entries'] != full_vocab_size:
                        raise ValueError('Entropy request/prefix/vocabulary mismatch')
                else:
                    generated = output.outputs[0]
                    value = dict(token_ids=list(generated.token_ids),
                                 finish_reason=generated.finish_reason)
                value['worker_batch_seconds'] = batch_seconds
                value['worker_batch_size'] = len(pending)
                value['queue_wait_seconds'] = queue_wait
                value['postprocess_seconds'] = time.monotonic() - post_started
                publish(base, path, value)
        except Exception as error:
            for path, _, _, _ in pending:
                target = base / 'results' / path.name
                if not target.exists():
                    publish(base, path, {'error': f'{type(error).__name__}: {error}'})
            raise


if __name__ == '__main__':
    main()
