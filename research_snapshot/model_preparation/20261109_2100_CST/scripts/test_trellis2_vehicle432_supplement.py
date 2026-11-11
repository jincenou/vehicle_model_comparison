import ast
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import dispatch_trellis2_vehicle432_supplement as worker

class SupplementTests(unittest.TestCase):
    def test_actual_adapter_claim_precedes_inference(self):
        tree=ast.parse(worker.ADAPTER.read_text())
        branch=next(x for x in tree.body if isinstance(x,ast.If) and ast.unparse(x.test)=='args.reload_only')
        first=branch.orelse[0]
        self.assertEqual(ast.unparse(first),'O.mkdir(parents=True, exist_ok=False)')
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and 'from_pretrained' in ast.unparse(n.func)]
        self.assertTrue(calls);self.assertTrue(all(n.lineno>first.lineno for n in calls))

    def test_atomic_directory_race_generates_once(self):
        with tempfile.TemporaryDirectory() as temp:
            dest=Path(temp)/'asset'
            program='from pathlib import Path; import sys; p=Path(sys.argv[1]); p.mkdir(exist_ok=False); (p/"generated").write_text("once")'
            a=subprocess.Popen([sys.executable,'-c',program,str(dest)],stderr=subprocess.DEVNULL)
            b=subprocess.Popen([sys.executable,'-c',program,str(dest)],stderr=subprocess.DEVNULL)
            self.assertEqual(sorted([a.wait(),b.wait()]),[0,1])
            self.assertEqual((dest/'generated').read_text(),'once')

    def test_supplement_success_reconciles_legacy_claim_skip(self):
        old=[dict(sample_id='a',status='failed',reason="FileExistsError(17, 'File exists')")]
        new=[dict(sample_id='a',status='completed',output_sha256='verified')]
        actual=worker.effective_jobs(old,new)[0]
        self.assertEqual(actual['status'],'completed');self.assertTrue(actual['legacy_existing_output_skip'])
        self.assertEqual(old[0]['status'],'failed')

    def test_delegation_keeps_legacy_result_and_preserves_population(self):
        old=[dict(sample_id='a',status='completed'),dict(sample_id='b',status='failed',reason='native error')]
        new=[dict(sample_id='a',status='delegated_to_legacy')]
        actual=worker.effective_jobs(old,new)
        self.assertEqual([r['status'] for r in actual],['completed','failed'])
        self.assertEqual(len(actual),2)

    def test_running_and_real_supplement_failure_are_not_successes(self):
        old=[dict(sample_id='a',status='queued'),dict(sample_id='b',status='queued')]
        new=[dict(sample_id='a',status='running'),dict(sample_id='b',status='failed',reason='empty native prediction')]
        self.assertEqual([x['status'] for x in worker.effective_jobs(old,new)],['running','failed'])

if __name__=='__main__':unittest.main()
