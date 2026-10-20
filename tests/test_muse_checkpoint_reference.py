"""Independent checkpoint oracle against the separately constructed tiny graph."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
from muse_graph_reference import checkpoint, reference

class CheckpointReferenceTests(unittest.TestCase):
    def check_graph(self,packed):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);model=root/'tiny.gguf';ids=root/'ids.i32';out=root/'logits.f32'
            weights=checkpoint(model,packed=packed);np.arange(1,11,dtype='<i4').tofile(ids)
            expected=reference(weights)[-1]
            run=subprocess.run([sys.executable,str(Path(__file__).with_name('muse_checkpoint_reference.py')),
                '--model',str(model),'--prompt-ids',str(ids),'--output',str(out),
                '--experts-per-token','2','--window','4'],capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr)
            actual=np.fromfile(out,dtype='<f4')
            self.assertLess(float(np.max(np.abs(actual-expected))),5e-5)
            self.assertEqual(int(actual.argmax()),int(expected.argmax()))

    def test_packed_bounded_reader_matches_independent_graph(self):
        self.check_graph(True)

    def test_f32_bounded_reader_matches_independent_graph(self):
        self.check_graph(False)

if __name__=='__main__':unittest.main()
