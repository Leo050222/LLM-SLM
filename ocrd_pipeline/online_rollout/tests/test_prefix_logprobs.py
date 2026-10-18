import unittest
from online_rollout.prefix_logprobs import PrefixLogprobs

def pack(entry,token):return (entry[token],[token],[entry[token]])
class Tests(unittest.TestCase):
    def test_compact_alignment(self):
        c=PrefixLogprobs();c.ingest_prompt([1,2,3],[None,{2:-2},{3:-3}],0,pack)
        c.ingest_prompt([1,2,3,4,5],[None,{4:-4},{5:-5}],2,pack)
        self.assertEqual(c.get_scores([1],[2,3,4,5])['chosen'],[-2,-3,-4,-5])
    def test_branch_isolation(self):
        c=PrefixLogprobs();c.ingest_generation([1],[2,3],[{2:-2},{3:-3}],pack)
        self.assertIsNone(c.get_scores([1,9],[3]))
        self.assertEqual(c.get_scores([1],[2,3])['chosen'],[-2,-3])
    def test_full_layout_ignores_cached_garbage(self):
        c=PrefixLogprobs();c.ingest_prompt([1,2,3,4],[None,{}, {},{4:-4}],2,pack)
        self.assertEqual(c.get_scores([1,2,3],[4])['chosen'],[-4])
        self.assertIsNone(c.get_scores([1],[2]))
    def test_incompatible_layout_fails(self):
        with self.assertRaises(ValueError):PrefixLogprobs().ingest_prompt([1,2,3],[None],0,pack)
    def test_boundary_needs_repair_without_logits(self):
        c=PrefixLogprobs()
        c.ingest_prompt([1,2,3,4],[None,{4:-4}],2,pack)
        self.assertEqual(c.missing_positions([1,2],[3,4]),[2])
        c.ingest_generation([1,2],[3],[{3:-3}],pack)
        self.assertEqual(c.get_scores([1,2],[3,4])['chosen'],[-3,-4])
if __name__=='__main__':unittest.main()
