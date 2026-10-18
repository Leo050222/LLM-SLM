"""Process-local raw-logit entropy capture; shared vLLM installation is untouched.

Only explicitly bracketed, one-token entropy batches activate capture. Ordinary
sampling/scoring always delegates unchanged. Request IDs and positions survive
chunked prefill and scheduler reordering; only the final prefix position is used.
"""
import torch
from entropy_mapping import external_records
from vllm.v1.worker.gpu.sample import logprob, sampler

_original_gather = logprob.compute_token_logprobs
_original_sample = sampler.Sampler.__call__
_active = False
_validate = False
_records = {}


def compute_token_logprobs(logits, token_ids):
    # Reference-only path used by the real-model numerical witness.
    if token_ids.shape[-1] <= 4096:
        return _original_gather(logits, token_ids)
    return torch.log_softmax(logits.float(), dim=-1).gather(-1, token_ids.long())


def full_entropy(logits):
    # Match the old FP32 log-softmax followed by FP64 exp and reduction exactly,
    # while keeping the vocabulary tensor on GPU and avoiding sorting/Python dicts.
    lp = torch.log_softmax(logits.float(), dim=-1).double()
    p = lp.exp()
    h = -(p * lp.masked_fill(~torch.isfinite(lp), 0)).sum(-1)
    return torch.stack((h, p.sum(-1)), dim=-1)


def capture_sample(self, logits, input_batch):
    if _active:
        if self.logprobs_mode != 'raw_logprobs' or input_batch.num_draft_tokens:
            raise RuntimeError('Entropy requires raw, non-speculative logits')
        if logits.shape[0] != len(input_batch.req_ids):
            raise RuntimeError('Unexpected entropy request/logit mapping')
        values = full_entropy(logits).cpu().tolist()
        references = None
        if _validate:
            import numpy as np
            lp_cpu = torch.log_softmax(logits.float(), dim=-1).cpu().numpy().astype(np.float64)
            finite = np.isfinite(lp_cpu)
            references = -(np.exp(lp_cpu) * np.where(finite, lp_cpu, 0)).sum(-1)
            for (h, _), reference in zip(values, references, strict=True):
                if abs(h - float(reference)) >= 1e-7:
                    raise ValueError(('Same-logit CPU/GPU entropy mismatch', h, float(reference)))
        positions = input_batch.positions[input_batch.logits_indices].cpu().tolist()
        for row, (key, pos, (entropy, mass)) in enumerate(zip(input_batch.req_ids, positions, values, strict=True)):
            if not .999 < mass < 1.001 or not 0 <= entropy <= 20:
                raise ValueError('Invalid full-vocabulary entropy')
            _records[key] = dict(entropy=entropy, probability_mass=mass,
                                 vocabulary_entries=logits.shape[-1], full_vocabulary=True,
                                 position=pos)
            if references is not None:
                _records[key]['same_logits_cpu_reference'] = float(references[row])
    return _original_sample(self, logits, input_batch)


logprob.compute_token_logprobs = compute_token_logprobs
sampler.Sampler.__call__ = capture_sample


class EntropyWorkerExtension:
    def begin_entropy_capture(self, validate=False):
        global _active, _validate
        if _active:
            raise RuntimeError('Nested entropy capture')
        _records.clear()
        _active = True
        _validate = validate
        return True

    def end_entropy_capture(self):
        global _active
        _active = False
        result = external_records(_records)
        _records.clear()
        return result
