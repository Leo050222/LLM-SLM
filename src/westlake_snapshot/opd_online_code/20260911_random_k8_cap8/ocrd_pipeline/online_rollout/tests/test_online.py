import copy
import sys
import tempfile
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from online_rollout.core import Settings, rollout, FileRPC, atomic_json, digest


ROW = dict(dataset_index=0, problem='question', prompt_token_ids=[99])


class Fake:
    max_model_len = 100
    def __init__(self, anchors=()):
        self.anchors = list(anchors)
        self.calls = []
    def generate(self, role, prefix, n, seed, banned_first=None):
        self.calls.append((role, list(prefix), n, banned_first))
        return dict(token_ids=list(range(10, 10 + n)) if role == 'student' else [77] * n,
                    finish_reason='length')
    def candidates(self, prefix, ids, threshold):
        position = self.anchors.pop(0) if self.anchors else None
        return [] if position is None else [dict(response_position=position, student_token_id=ids[position])]


class OnlineTests(unittest.TestCase):
    def test_multiple_takeovers_new_prefix_and_budget(self):
        backend = Fake([1, 0, None])
        prefixes = []
        def judge(row, prefix, candidate):
            prefixes.append(prefix)
            return {'select_anchor': True}
        state = rollout(ROW, Settings(chunk_tokens=4, teacher_tokens=2, max_tokens=9), backend, judge)
        self.assertEqual(prefixes, [[99, 10], [99, 10, 77, 77]])
        self.assertEqual(state['response_token_ids'], [10, 77, 77, 77, 77, 10, 11, 12, 13])
        self.assertEqual(state['discarded_tokens'], 7)
        self.assertEqual(state['token_sources'].count('teacher'), 4)
        self.assertEqual(state['judge_calls'], 2)
        self.assertEqual(state['finish_reason'], 'length')

    def test_no_takeover_count_cap(self):
        backend = Fake([0] * 10)
        state = rollout(ROW, Settings(max_tokens=80), backend,
                        lambda *args: {'select_anchor': True})
        self.assertEqual(len(state['events']), 10)
        self.assertEqual(state['token_sources'], ['teacher'] * 80)
        self.assertEqual(state['finish_reason'], 'length')

    def test_no_anchor_keeps_every_token(self):
        result = rollout(ROW, Settings(chunk_tokens=4, max_tokens=7), Fake(), None)
        self.assertEqual(len(result['response_token_ids']), 7)
        self.assertEqual(result['discarded_tokens'], 0)

    def test_judge_no_then_yes_stops_before_later_candidate(self):
        backend = Fake()
        backend.candidates = lambda prefix, ids, threshold: [
            dict(response_position=i, student_token_id=ids[i]) for i in range(len(ids))]
        seen = []
        def judge(row, prefix, candidate):
            seen.append(candidate['response_position'])
            return dict(select_anchor=candidate['response_position'] == 1)
        result = rollout(ROW, Settings(chunk_tokens=4, teacher_tokens=8, max_tokens=4), backend, judge)
        self.assertEqual(seen, [0, 1])
        self.assertEqual(result['response_token_ids'], [10, 77, 77, 77])

    def test_resume_matches_uninterrupted(self):
        settings = Settings(chunk_tokens=4, max_tokens=10)
        saved = []
        def save(state):
            saved.append(copy.deepcopy(state))
            raise RuntimeError('interrupted after commit')
        with self.assertRaises(RuntimeError):
            rollout(ROW, settings, Fake(), None, save=save)
        resumed = rollout(ROW, settings, Fake(), None, checkpoint=saved[0])
        expected = rollout(ROW, settings, Fake(), None)
        self.assertEqual(resumed, expected)
        with self.assertRaises(ValueError):
            rollout(ROW, Settings(max_tokens=11), Fake(), None, checkpoint=resumed)

    def test_eos_stops_student_and_teacher(self):
        for anchor in (None, 0):
            backend = Fake([anchor])
            original = backend.generate
            def generate(role, prefix, n, seed, banned_first=None):
                value = original(role, prefix, n, seed, banned_first)
                if (anchor is None and role == 'student') or role == 'teacher':
                    value.update(token_ids=value['token_ids'][:1], finish_reason='stop')
                return value
            backend.generate = generate
            state = rollout(ROW, Settings(max_tokens=10), backend, lambda *a: {'select_anchor': True})
            self.assertEqual(len(state['response_token_ids']), 1)
            self.assertEqual(state['finish_reason'], 'stop')

    def test_context_boundary(self):
        backend = Fake()
        backend.max_model_len = 5
        result = rollout(ROW, Settings(), backend, None)
        self.assertEqual(len(result['response_token_ids']), 3)
        self.assertEqual(result['finish_reason'], 'context_limit')

    def test_bad_bridge_rejected(self):
        backend = Fake([0])
        original = backend.generate
        def generate(role, prefix, n, seed, banned_first=None):
            value = original(role, prefix, n, seed, banned_first)
            if role == 'teacher':
                value['token_ids'][0] = banned_first
            return value
        backend.generate = generate
        with self.assertRaises(ValueError):
            rollout(ROW, Settings(max_tokens=10), backend, lambda *a: {'select_anchor': True})

    def test_rpc_cache_and_error(self):
        with tempfile.TemporaryDirectory() as root:
            payload = {'op': 'generate'}
            path = Path(root) / 'student' / 'results' / (digest(payload) + '.json')
            atomic_json(path, {'token_ids': [1]})
            self.assertEqual(FileRPC(root).call('student', payload)['token_ids'], [1])
            atomic_json(path, {'error': 'test'})
            with self.assertRaises(RuntimeError):
                FileRPC(root).call('student', payload)


if __name__ == '__main__':
    unittest.main()
