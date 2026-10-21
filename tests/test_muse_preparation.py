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
            self.assertIn(str(repo.resolve()/'tests/muse_live_tools.py'),plan['api_multiple_tools'])
            self.assertIn('--output',plan['api_multiple_tools'])
            self.assertIn('muse_live_tools.py',(root/'RUNBOOK.md').read_text())
            reasoning=plan['agent_reasoning']
            self.assertEqual(reasoning[reasoning.index('--temperature')+1],'1')
            self.assertEqual(reasoning[reasoning.index('--tokens')+1],'4096')
            self.assertIn('--require-compaction-cache',reasoning)
            self.assertIn('--compaction-only',plan['agent_cache'])
            self.assertIn('--require-compaction-cache',plan['agent_cache'])
            self.assertEqual(plan['server'][plan['server'].index('--prefill-chunk')+1],'512')
            baseline=next(b for b in plan['benchmarks'] if b['name']=='auto-chunk512')
            self.assertEqual(baseline['command'][baseline['command'].index('--prefill-chunk')+1],'512')
            self.assertGreater(len((root/'benchmark.txt').read_text().split()),16384)
            self.assertTrue((root/'RUNBOOK.md').exists())
            with self.assertRaises(FileExistsError):prepare(root,repo,model)

if __name__=='__main__':unittest.main()
