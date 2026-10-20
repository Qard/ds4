import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'gguf-tools'))
from qwen4_pack_to_qwen4exp import Reader, kv_bytes, w_str
from gguf_join import join_gguf


def shard(path, index, tensors):
    kv = [('general.architecture', 8, 'muse-spark'), ('split.no', 2, index),
          ('split.count', 2, 2), ('split.tensors.count', 5, 2)]
    data, infos = bytearray(), bytearray()
    for name, values in tensors:
        offset = (len(data) + 31) // 32 * 32
        data.extend(b'\0' * (offset - len(data)))
        data.extend(struct.pack('<' + 'f' * len(values), *values))
        infos.extend(w_str(name) + struct.pack('<IQIQ', 1, len(values), 0, offset))
    header = b'GGUF' + struct.pack('<IQQ', 3, len(tensors), len(kv))
    header += b''.join(kv_bytes(*entry) for entry in kv) + infos
    path.write_bytes(header + b'\0' * (-len(header) % 32) + data)


class JoinTests(unittest.TestCase):
    def test_join_preserves_tensor_bytes_and_removes_split_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a, b, out = root / 'a.gguf', root / 'b.gguf', root / 'out.gguf'
            shard(a, 0, [('a.weight', [1., 2.])])
            shard(b, 1, [('b.weight', [3., 4., 5.])])
            join_gguf([a, b], out)
            r = Reader(str(out))
            try:
                self.assertFalse(any(k.startswith('split.') for k in r.kv))
                for name, expected in [('a.weight', (1., 2.)), ('b.weight', (3., 4., 5.))]:
                    kind, dims, offset = r.tensors[name]
                    r.f.seek(r.data_start + offset)
                    self.assertEqual(struct.unpack('<' + 'f' * dims[0], r.f.read(4 * dims[0])), expected)
            finally:
                r.f.close()
            self.assertTrue(a.exists() and b.exists())
            with self.assertRaises(FileExistsError):
                join_gguf([a, b], out)

    def test_failed_second_source_resumes_without_recopying_first_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a, b, out = root / 'a.gguf', root / 'b.gguf', root / 'out.gguf'
            shard(a, 0, [('a.weight', [1., 2.])])
            shard(b, 1, [('b.weight', [3., 4.])])
            def interrupted(index):
                if index == 1:
                    raise OSError('interrupted download')
                return a
            with self.assertRaisesRegex(OSError, 'interrupted download'):
                join_gguf([a, b], out, acquire=interrupted)
            seen = []
            def resumed(index):
                seen.append(index)
                return [a, b][index]
            join_gguf([a, b], out, acquire=resumed)
            self.assertEqual(seen, [1])

    def test_resume_rejects_journal_boundary_inside_tensor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a, b, out = root / 'a.gguf', root / 'b.gguf', root / 'out.gguf'
            shard(a, 0, [('a.weight', [1., 2.])])
            shard(b, 1, [('b.weight', [3., 4.])])
            def interrupted(index):
                if index == 1:
                    raise OSError('interrupted download')
                return a
            with self.assertRaises(OSError):
                join_gguf([a, b], out, acquire=interrupted)
            journal = Path(str(out) + '.progress.json')
            state = json.loads(journal.read_text())
            state['bytes'] -= 1
            journal.write_text(json.dumps(state))
            with self.assertRaisesRegex(ValueError, 'boundary'):
                join_gguf([a, b], out)

    def test_duplicate_tensor_is_rejected_before_output_is_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a, b, out = root / 'a.gguf', root / 'b.gguf', root / 'out.gguf'
            shard(a, 0, [('same.weight', [1.])])
            shard(b, 1, [('same.weight', [2.])])
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                join_gguf([a, b], out)
            self.assertFalse(out.exists())


if __name__ == '__main__':
    unittest.main()
