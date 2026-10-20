import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from muse_native_validation import compare_runs


class NativeValidationTests(unittest.TestCase):
    def write_run(self, root, logits, ids=(2, 3), prompt=(1, 2)):
        root.mkdir()
        (root / 'summary.json').write_text(json.dumps({'cases': [
            {'name': 'raw', 'vocab': 4, 'steps': 2, 'prompt_tokens': len(prompt)}]}))
        np.asarray(logits, dtype='<f4').tofile(root / 'raw.logits.f32')
        np.asarray(ids, dtype='<i4').tofile(root / 'raw.ids.i32')
        np.asarray(prompt, dtype='<i4').tofile(root / 'raw.prompt.i32')

    def test_all_vectors_and_same_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / 'a', Path(tmp) / 'b'
            x = np.arange(12).reshape(3, 4)
            self.write_run(a, x)
            self.write_run(b, x + .0001)
            report = compare_runs(a, b, .002)
            self.assertLess(report[0]['max_error'], .0002)
            self.assertEqual(report[0]['argmax_matches'], 3)
            np.array([2, 1], dtype='<i4').tofile(b / 'raw.ids.i32')
            with self.assertRaisesRegex(ValueError, 'continuation'):
                compare_runs(a, b, .002)

    def test_empty_summary_cannot_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'summary.json').write_text('{"cases": []}')
            with self.assertRaisesRegex(ValueError,'empty'):
                compare_runs(root,root,.002)

    def test_bad_artifacts_fail(self):
        for bad in ('nonfinite', 'truncated', 'drift', 'prompt', 'partialbyte'):
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as tmp:
                a, b = Path(tmp) / 'a', Path(tmp) / 'b'
                x = np.arange(12, dtype=np.float32).reshape(3, 4)
                self.write_run(a, x)
                if bad == 'nonfinite': x[2, 1] = np.nan
                if bad == 'truncated': x = x[:-1]
                if bad == 'drift': x[1, 2] += .1
                self.write_run(b, x, prompt=(1, 3) if bad == 'prompt' else (1, 2))
                if bad == 'partialbyte':
                    with (b / 'raw.logits.f32').open('ab') as fp:
                        fp.write(b'x')
                with self.assertRaises(ValueError):
                    compare_runs(a, b, .002)


if __name__ == '__main__':
    unittest.main()
