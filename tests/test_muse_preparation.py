import json
import tempfile
import unittest
from pathlib import Path
from muse_prepare_validation import prepare

class PreparationTests(unittest.TestCase):
    def test_preparation_never_needs_a_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'prepared';repo=Path(tmp)/'repo with spaces';model=Path(tmp)/'missing.gguf'
            prepare(root,repo,model)
            plan=json.loads((root/'commands.json').read_text())
            self.assertEqual(plan['model'],str(model.resolve()))
            self.assertEqual(len(plan['benchmarks']),3)
            self.assertIn(str(repo.resolve()/'ds4-bench'),plan['benchmarks'][0]['command'])
            self.assertIn('--teacher-forced-decode',plan['benchmarks'][0]['command'])
            self.assertNotEqual(plan['server'][-1],plan['server_restart'][-1])
            self.assertEqual(plan['server'][:-1],plan['server_restart'][:-1])
            self.assertIn('--require-initial-cache',plan['api_restored'])
            reasoning=plan['agent_reasoning']
            self.assertEqual(reasoning[reasoning.index('--temperature')+1],'1')
            self.assertGreater(len((root/'benchmark.txt').read_text().split()),16384)
            self.assertTrue((root/'RUNBOOK.md').exists())
            with self.assertRaises(FileExistsError):prepare(root,repo,model)

if __name__=='__main__':unittest.main()
