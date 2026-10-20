"""Small files exercise the same byte ranges used by the 346 GiB checkpoint."""

import json
import struct
import tempfile
import unittest
from pathlib import Path

from muse.storage import ExpertCache, TensorStore


def write_shard(path, tensors):
    header, data = {}, bytearray()
    for name, (dtype, shape, payload) in tensors.items():
        start = len(data)
        data.extend(payload)
        header[name] = dict(dtype=dtype, shape=shape, data_offsets=[start, len(data)])
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_expert_slice_reads_only_selected_rows_across_shards(self):
        names = [f"model.language_model.layers.0.mlp.experts.{p}.weight"
                 for p in ("gate_proj", "up_proj", "down_proj")]
        for i, name in enumerate(names):
            write_shard(self.root / f"{i}.safetensors", {
                name: ("U32", [3, 2], struct.pack("<6I", *range(6)))})
        (self.root / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {name: f"{i}.safetensors" for i, name in enumerate(names)}}))
        with TensorStore(self.root) as store:
            self.assertEqual(store.read(names[0], row=2), struct.pack("<2I", 4, 5))
            self.assertEqual(store.read_bytes, 8)
            self.assertEqual(store.expert_bytes(0), 24)
            self.assertEqual(len(store.expert_tensors(0)), 3)
            with self.assertRaises(IndexError):
                store.read(names[0], row=3)
            with self.assertRaises(IndexError):
                store.read(names[0], row=-1)

    def test_rejects_tensor_extent_inconsistent_with_shape(self):
        write_shard(self.root / "model.safetensors", {
            "bad": ("F32", [8], b"1234")})
        with self.assertRaisesRegex(ValueError, "shape|extent"):
            TensorStore(self.root)

    def test_rejects_truncated_file_before_inference(self):
        write_shard(self.root / "model.safetensors", {
            "bad": ("F32", [1], b"1234")})
        path = self.root / "model.safetensors"
        path.write_bytes(path.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, "truncated|extent"):
            TensorStore(self.root)

    def test_rejects_shard_path_escape(self):
        (self.root / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {"bad": "../outside.safetensors"}}))
        with self.assertRaisesRegex(ValueError, "shard path"):
            TensorStore(self.root)


class CacheTests(unittest.TestCase):
    def test_hit_and_eviction_under_byte_budget(self):
        reads = []
        def load(key):
            reads.append(key)
            return bytes([key]) * 4
        with ExpertCache(8, lambda _: 4, load, workers=2) as cache:
            with cache.lease([1, 2]) as values:
                self.assertEqual(values[1], b"\x01" * 4)
            with cache.lease([2, 3]):
                pass
            self.assertEqual(cache.bytes_used, 8)
            self.assertEqual(cache.stats["hits"], 1)
            self.assertEqual(cache.stats["misses"], 3)
            self.assertEqual(cache.stats["evictions"], 1)
            self.assertCountEqual(reads, [1, 2, 3])

    def test_oversized_lease_rejected_before_any_io(self):
        with ExpertCache(7, lambda _: 4, lambda _: self.fail("unexpected I/O")) as cache:
            with self.assertRaisesRegex(ValueError, "budget"):
                with cache.lease([1, 2]):
                    pass
            self.assertEqual(cache.bytes_used, 0)

    def test_nested_lease_cannot_evict_in_use_experts(self):
        with ExpertCache(4, lambda _: 4, lambda k: k) as cache:
            with cache.lease([1]):
                with self.assertRaisesRegex(RuntimeError, "lease"):
                    with cache.lease([2]):
                        pass

    def test_io_failure_does_not_leave_failed_expert_cached(self):
        def load(key):
            if key == 2:
                raise OSError("short read")
            return key
        with ExpertCache(8, lambda _: 4, load) as cache:
            with self.assertRaisesRegex(OSError, "short read"):
                with cache.lease([1, 2]):
                    pass
            self.assertLessEqual(cache.bytes_used, 8)
            with cache.lease([1]) as values:
                self.assertEqual(values[1], 1)


if __name__ == "__main__":
    unittest.main()
