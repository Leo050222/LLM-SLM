import tempfile, unittest, queue, threading
from pathlib import Path
from deadline import Budget, TraceDeadline, consume_tasks

class Tests(unittest.TestCase):
    def test_cumulative_budget_excludes_job_downtime(self):
        with tempfile.TemporaryDirectory() as d:
            now = [0.0]; p = Path(d) / 'time.json'
            b = Budget(p, 900, lambda: now[0])
            now[0] = 899; b.check(); b.persist()
            now[0] = 5000
            resumed = Budget(p, 900, lambda: now[0])
            self.assertEqual(resumed.elapsed(), 899)
            now[0] += 1
            with self.assertRaises(TraceDeadline): resumed.check()
            self.assertEqual(resumed.elapsed(), 900)

    def test_next_group_runs_while_prior_tail_waits(self):
        q = queue.Queue()
        for task in [('a','tail'), ('a','short'), ('b','next')]: q.put(task)
        entered = threading.Event(); later = threading.Event(); done = []
        def action(task, replica):
            if task[1] == 'tail':
                entered.set(); self.assertTrue(later.wait(2))
            elif task[0] == 'b':
                self.assertTrue(entered.is_set()); later.set()
            done.append(task)
        a = threading.Thread(target=consume_tasks, args=(q,action,0));a.start()
        self.assertTrue(entered.wait(2))
        b = threading.Thread(target=consume_tasks, args=(q,action,1));b.start();a.join();b.join()
        self.assertEqual(len(done),3)
        self.assertLess(done.index(('b','next')),done.index(('a','tail')))

if __name__ == '__main__': unittest.main()
