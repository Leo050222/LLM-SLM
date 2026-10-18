"""CPU tests for causal alignment and frozen event identity, no model dependencies."""
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from run import freeze_trace, chunk_selected


def fixture():
    candidate = dict(student_token_id=9, signed_logprob_gap_student_minus_teacher=3.,
                     absolute_sequence_position=3, student_logprob=-.2,
                     teacher_logprob_on_student_token=-3.2,
                     teacher_top_tokens=[dict(token_id=7, logprob=-.1)])
    return dict(status='COMPLETE', prompt_token_ids=[1,2], response_token_ids=[3,7,8],
                token_sources=['student','teacher','teacher'], dataset='test', problem_id='p',
                dataset_index=0, finish_reason='stop', events=[
                    dict(anchor_response_position=1,bridge_token_ids=[7],first_token_ban=True,
                         student_entry_token=9,selected_candidate=candidate),
                    dict(anchor_response_position=2,bridge_token_ids=[8],first_token_ban=True,
                         student_entry_token=9,selected_candidate={**candidate,'absolute_sequence_position':4})])


class AlignmentTests(unittest.TestCase):
    def freeze(self, x):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'trace.json';p.write_text(json.dumps(x))
            return freeze_trace(x,p,7,'code',100)

    def test_adjacent_events_and_terminal_token(self):
        r=self.freeze(fixture())
        self.assertEqual(r['input_ids'],[1,2,3,7])
        self.assertEqual([e['hidden_position'] for e in r['events']],[2,3])
        self.assertEqual([e['event_id'] for e in r['events']],[100,101])
        for e,expected in zip(r['events'],[[1,2,3],[1,2,3,7]]):
            self.assertEqual(e['prefix_ids_u32le_sha256'],hashlib.sha256(struct.pack('<'+'I'*len(expected),*expected)).hexdigest())

    def test_chunk_boundaries(self):
        events=[dict(hidden_position=i) for i in [0,511,512,1023,1024]]
        self.assertEqual(list(chunk_selected(1025,events)),[
            (0,512,[(0,0),(1,511)]),(512,1024,[(2,0),(3,511)]),(1024,1025,[(4,0)])])

    def test_no_events(self):
        x=fixture();x['events']=[];x['token_sources']=['student']*3
        r=self.freeze(x);self.assertEqual(r['events'],[]);self.assertEqual(r['input_ids'],[])

    def test_wrong_absolute_offset_rejected(self):
        x=fixture();x['events'][0]['selected_candidate']['absolute_sequence_position']=4
        with self.assertRaises(AssertionError):self.freeze(x)

    def test_same_token_rejected(self):
        x=fixture();x['events'][0]['student_entry_token']=7
        with self.assertRaises(AssertionError):self.freeze(x)

    def test_missing_event_rejected(self):
        x=fixture();x['events'].pop()
        with self.assertRaises(AssertionError):self.freeze(x)

    def test_threshold_equality_rejected(self):
        x=fixture();x['events'][0]['selected_candidate']['signed_logprob_gap_student_minus_teacher']=2
        with self.assertRaises(AssertionError):self.freeze(x)

    def test_first_response_position(self):
        x=fixture();x['events']=x['events'][:1];x['response_token_ids']=[7]
        x['token_sources']=['teacher'];x['events'][0]['anchor_response_position']=0
        x['events'][0]['selected_candidate']['absolute_sequence_position']=2
        r=self.freeze(x);self.assertEqual(r['input_ids'],[1,2]);self.assertEqual(r['events'][0]['hidden_position'],1)


if __name__=='__main__':unittest.main()
