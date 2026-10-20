import json
import tempfile
import unittest
from pathlib import Path
from muse_compare_frontiers import compare_frontiers

class FrontierTests(unittest.TestCase):
    def test_same_checkpoint_and_all_frontiers_are_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/'a',Path(tmp)/'b';a.mkdir();b.mkdir()
            data=dict(model='/models/muse.gguf',backend='metal',quality=False,quant_bits=2,
                      frontier_tokens=1024,ctx=8257,vocab=4,argmax_id=3,logits=[.1,.2,.3,.4])
            for root in (a,b):(root/'frontier_001024.logits.json').write_text(json.dumps(data))
            report=compare_frontiers(a,b,.002,frontiers=(1024,))
            self.assertEqual(report[0]['max_error'],0)
            with self.assertRaises(ValueError):compare_frontiers(a,b,.002)
            data['model']='/models/different.gguf'
            (b/'frontier_001024.logits.json').write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'model'):compare_frontiers(a,b,.002,frontiers=(1024,))

if __name__=='__main__':unittest.main()
