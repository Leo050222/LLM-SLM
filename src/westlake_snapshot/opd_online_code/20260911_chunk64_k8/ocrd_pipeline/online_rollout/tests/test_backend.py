"""Test real candidate-builder/RPC adapter wiring without GPU dependencies."""
import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from online_rollout.run import Backend


class Tokenizer:
    def convert_ids_to_tokens(self, value):
        return str(value)
    def decode(self, values, **kwargs):
        return ','.join(map(str, values))


class RPC:
    def __init__(self):
        self.calls = []
    def call(self, role, req):
        self.calls.append((role, req))
        if req['op'] == 'score':
            return dict(chosen=[-1., -1., -5.] if role == 'student' else [-3., -3.01, -1.],
                        top_ids=[[1, 2]] * 3, top_lp=[[-.5, -1.]] * 3)
        return dict(token_ids=[77] * req['n'], finish_reason='length')


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.rpc = RPC()
        self.backend = Backend(self.rpc, {r: {'max_model_len': 100} for r in ('student', 'teacher')}, Tokenizer())

    def test_exact_gap_boundary_and_both_directions(self):
        candidates = self.backend.candidates([99], [10, 11, 12], 2.)
        self.assertEqual([c['response_position'] for c in candidates], [1, 2])
        self.assertEqual([c['gap_direction'] for c in candidates], ['student_higher', 'teacher_higher'])
        self.assertTrue(all(c[1]['prefix'] == [99] for c in self.rpc.calls))

    def test_first_token_only_banned_and_exact_eight(self):
        result = self.backend.generate('teacher', [99], 8, 42, banned_first=10)
        self.assertEqual(len(result['token_ids']), 8)
        first, second = [r for _, r in self.rpc.calls]
        self.assertEqual(first['n'], 1)
        self.assertEqual(first['banned_first'], 10)
        self.assertNotIn('banned_first', second)
        self.assertEqual(second['prefix'], [99, 77])
        self.assertEqual(second['n'], 7)


if __name__ == '__main__':
    unittest.main()
