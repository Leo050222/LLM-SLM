"""Per-job vLLM compatibility patch; never edits shared site-packages.

The Triton gather kernel is designed for small top-k. For a full-vocab query
use the mathematically identical FP32 log_softmax + gather, linear in vocabulary.
"""
import torch
from vllm.v1.worker.gpu.sample import logprob

_original = logprob.compute_token_logprobs

def compute_token_logprobs(logits, token_ids):
    if token_ids.shape[-1] <= 4096:
        return _original(logits, token_ids)
    return torch.log_softmax(logits.float(), dim=-1).gather(-1, token_ids.long())

logprob.compute_token_logprobs = compute_token_logprobs

class EntropyWorkerExtension:
    """Importing this worker extension installs the scoped large-K fallback."""
    pass
