"""Model-local logprob sidecar for vLLM's content-addressed KV cache.

Keys include every prefix token, so rollback and diverging continuations cannot
reuse a probability from the discarded branch. KV blocks remain owned by vLLM.
"""
import hashlib
import struct


def prefix_keys(tokens):
    h = hashlib.sha256(b'opd-logprobs-v1')
    keys = []
    for token in tokens:
        h.update(struct.pack('<Q', token))
        keys.append(h.digest())
    return keys


class PrefixLogprobs:
    def __init__(self):
        self.records = {}

    def put(self, tokens, position, record):
        self.records[prefix_keys(tokens[:position + 1])[-1]] = record

    def get_scores(self, prefix, tokens):
        keys = prefix_keys(prefix + tokens)[len(prefix):]
        if any(key not in self.records for key in keys):
            return None
        rows = [self.records[key] for key in keys]
        return dict(chosen=[r[0] for r in rows], top_ids=[r[1] for r in rows],
                    top_lp=[r[2] for r in rows])

    def ingest_prompt(self, ids, entries, cached_tokens, pack):
        # V2 emits only uncached prompt logprobs, preceded by None. Older
        # runners may return a full-sized array with uncomputed cache slots.
        if entries is None:
            if cached_tokens < len(ids)-1:
                raise ValueError('Missing uncached prompt logprobs')
            return
        if len(entries) == len(ids):
            shift = 0
        elif len(entries) == len(ids)-cached_tokens:
            shift = cached_tokens
        else:
            raise ValueError('Unknown cached prompt-logprob layout')
        keys = prefix_keys(ids)
        for pos in range(cached_tokens + 1, len(ids)):
            entry = entries[pos-shift]
            if entry is None:
                raise ValueError('Uncomputed logprob outside cached prefix')
            self.records[keys[pos]] = pack(entry, ids[pos])

    def ingest_generation(self, prefix, tokens, entries, pack):
        if entries is None or len(entries) != len(tokens):
            raise ValueError('Missing generation logprobs')
        keys = prefix_keys(prefix + tokens)
        for pos, entry in enumerate(entries, len(prefix)):
            self.records[keys[pos]] = pack(entry, tokens[pos-len(prefix)])
