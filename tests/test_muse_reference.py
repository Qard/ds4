import tempfile
import unittest
from pathlib import Path
import numpy as np
from muse_graph_reference import checkpoint, reference

class QuantizedReferenceTests(unittest.TestCase):
    def test_packed_graph_is_independently_evaluable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'packed.gguf'
            weights=checkpoint(path,packed=True)
            self.assertEqual(weights['blk.0.ffn_gate_exps.weight'].shape,(4,256,256))
            logits=reference(weights)
            self.assertEqual(logits.shape,(10,32))
            self.assertTrue(np.isfinite(logits).all())
            self.assertGreater(float(np.ptp(logits)),.1)
            self.assertLess(path.stat().st_size,4*1024**2)

if __name__=='__main__': unittest.main()
