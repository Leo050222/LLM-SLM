#!/usr/bin/env python3
"""Live worker contract witness; synthetic handoffs are NOT evaluation results."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from online_rollout.core import FileRPC, Settings, atomic_json, rollout
from online_rollout.run import Backend
from anchor_selector.core import tokenizer_fingerprint


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--queue', type=Path, required=True)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tokenizer', required=True)
    args = p.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    models = {role: json.loads((args.queue / role / 'model.json').read_text())
              for role in ('student', 'teacher')}
    assert all(m['tokenizer_sha256'] == tokenizer_fingerprint(tok) for m in models.values())
    backend = Backend(FileRPC(args.queue), models, tok)
    row = json.loads(args.input.read_text().splitlines()[0])
    prefix = row['prompt_token_ids']
    draft = backend.generate('student', prefix, 64, 12345)
    ids = draft['token_ids']
    assert len(ids) == 64
    alignment = {}
    for role in ('student', 'teacher'):
        batch = backend.call(role, op='score', prefix=prefix, tokens=ids)
        pos = 17
        single = backend.call(role, op='score', prefix=prefix + ids[:pos], tokens=ids[pos:pos+1])
        error = abs(batch['chosen'][pos] - single['chosen'][0])
        assert error < .05, (role, error)
        alignment[role] = error
    bridge = backend.generate('teacher', prefix + ids[:17], 8, 12346, banned_first=ids[17])
    assert len(bridge['token_ids']) == 8 and bridge['token_ids'][0] != ids[17]
    # Synthetic candidate at each chunk start forces two consecutive handoffs.
    backend.candidates = lambda prefix, ids, threshold: [dict(response_position=0, student_token_id=ids[0])]
    state = rollout(row, Settings(max_tokens=16), backend, lambda *a: dict(select_anchor=True))
    assert len(state['events']) == 2
    assert state['token_sources'] == ['teacher'] * 16
    result = dict(status='LIVE_WORKER_CONTRACT_OK', alignment_absolute_error=alignment,
                  bridge_tokens=len(bridge['token_ids']), synthetic_takeovers=2,
                  evaluation_result=False, models=models)
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
