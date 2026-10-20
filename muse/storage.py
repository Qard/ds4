"""Safetensors range reads and a byte-bounded, single-consumer expert cache.

No mmap, mx.load, or full expert-bank tensors: even an individual expert can
span several checkpoint shards. The cache owns loaded arrays until the caller
has evaluated all GPU work inside its lease.
"""

import json
import math
import os
import re
import struct
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


DTYPE_BYTES = {"BOOL": 1, "U8": 1, "I8": 1, "U16": 2, "I16": 2,
               "F16": 2, "BF16": 2, "U32": 4, "I32": 4, "F32": 4,
               "U64": 8, "I64": 8, "F64": 8}
EXPERT_RE = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\."
    r"(gate_proj|up_proj|down_proj)\.(weight|scales|biases)$")


@dataclass(frozen=True)
class TensorInfo:
    shard: str
    dtype: str
    shape: tuple
    offset: int
    nbytes: int

    @property
    def row_bytes(self):
        if not self.shape or not self.shape[0]:
            raise ValueError("Cannot slice a scalar or empty expert tensor")
        return self.nbytes // self.shape[0]


class TensorStore:
    """Validate headers up front; open/read weight payloads only when needed."""

    def __init__(self, root):
        self.root = Path(root)
        self.tensors = {}
        self._fds = {}
        self.read_bytes = 0
        self._expert_names = {}
        index = self.root / "model.safetensors.index.json"
        weight_map = json.loads(index.read_text())["weight_map"] if index.exists() else None
        shards = sorted(set(weight_map.values())) if weight_map is not None else [
            p.name for p in sorted(self.root.glob("model*.safetensors"))]
        if not shards:
            raise ValueError("No model safetensors or checkpoint index found")
        for shard in shards:
            path = Path(shard)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Invalid shard path: {shard}")
            with (self.root / shard).open("rb") as f:
                raw = f.read(8)
                if len(raw) != 8:
                    raise ValueError(f"Truncated safetensors header: {shard}")
                length = struct.unpack("<Q", raw)[0]
                size = os.fstat(f.fileno()).st_size
                if length > 32 * 1024**2 or length > size - 8:
                    raise ValueError(f"Invalid or truncated safetensors header: {shard}")
                header = json.loads(f.read(length))
            for name, entry in header.items():
                if name == "__metadata__":
                    continue
                dtype = entry["dtype"]
                shape = tuple(entry["shape"])
                offsets = entry["data_offsets"]
                if (dtype not in DTYPE_BYTES or len(offsets) != 2
                        or any(type(d) is not int or d < 0 for d in shape)
                        or any(type(o) is not int or o < 0 for o in offsets)):
                    raise ValueError(f"Invalid tensor descriptor: {name}")
                start, end = offsets
                nbytes = math.prod(shape) * DTYPE_BYTES[dtype]
                if end - start != nbytes or 8 + length + end > size:
                    raise ValueError(f"Invalid or truncated tensor extent/shape: {name}")
                if name in self.tensors:
                    raise ValueError(f"Duplicate tensor: {name}")
                if weight_map is not None and weight_map.get(name) != shard:
                    raise ValueError(f"Tensor disagrees with checkpoint index: {name}")
                self.tensors[name] = TensorInfo(shard, dtype, shape, 8 + length + start, nbytes)
                if match := EXPERT_RE.fullmatch(name):
                    self._expert_names.setdefault(int(match[1]), []).append(name)
        if weight_map is not None and set(weight_map) != set(self.tensors):
            raise ValueError("Missing tensors declared by checkpoint index")
        # Open once, before worker threads start, so no descriptor races occur.
        try:
            for shard in shards:
                self._fds[shard] = os.open(self.root / shard, os.O_RDONLY)
        except BaseException:
            self.close()
            raise

    def expert_tensors(self, layer):
        return tuple(sorted(self._expert_names.get(layer, ())))

    def expert_bytes(self, layer):
        return sum(self.tensors[name].row_bytes for name in self.expert_tensors(layer))

    def read(self, name, row=None):
        info = self.tensors[name]
        offset, size = info.offset, info.nbytes
        if row is not None:
            if not info.shape or not 0 <= row < info.shape[0]:
                raise IndexError(f"Expert row out of range: {name}: {row}")
            size = info.row_bytes
            offset += row * size
        # pread can return short reads, including after a model file changes.
        pieces, remaining = [], size
        while remaining:
            part = os.pread(self._fds[info.shard], remaining, offset)
            if not part:
                raise OSError(f"Short tensor read: {name}")
            pieces.append(part)
            offset += len(part)
            remaining -= len(part)
        self.read_bytes += size
        return pieces[0] if len(pieces) == 1 else b"".join(pieces)

    def close(self):
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class ExpertCache:
    """LRU across all layers; leases cannot exceed the explicit byte budget.

    Only one inference caller may hold a lease. Reads within that lease run in
    parallel. The size callback must include every cached projection and scale.
    Eviction happens *before* allocating replacements.
    """

    def __init__(self, budget, size, load, workers=4):
        if budget <= 0 or workers <= 0:
            raise ValueError("Cache budget and worker count must be positive")
        self.budget, self.size, self.load = budget, size, load
        self.bytes_used = 0
        self._entries = OrderedDict()
        self._leased = False
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="muse-ssd")
        self.stats = dict(hits=0, misses=0, evictions=0, read_bytes=0, read_seconds=0.0)

    @contextmanager
    def lease(self, keys):
        if self._leased:
            raise RuntimeError("An expert lease is already active")
        keys = tuple(dict.fromkeys(keys))
        sizes = {key: self.size(key) for key in keys}
        if any(size <= 0 for size in sizes.values()) or sum(sizes.values()) > self.budget:
            raise ValueError("Selected experts exceed cache budget")
        self._leased = True
        try:
            missing = [key for key in keys if key not in self._entries]
            required = sum(sizes[key] for key in missing)
            wanted = set(keys)
            for old in list(self._entries):
                if self.bytes_used + required <= self.budget:
                    break
                if old not in wanted:
                    _, size = self._entries.pop(old)
                    self.bytes_used -= size
                    self.stats["evictions"] += 1
            start = time.monotonic()
            futures = {key: self._pool.submit(self.load, key) for key in missing}
            try:
                # Drain every future even if one fails. No background allocation
                # may outlive a failed lease and overlap the next admission.
                error = None
                for key, future in futures.items():
                    try:
                        value = future.result()
                        self._entries[key] = (value, sizes[key])
                        self.bytes_used += sizes[key]
                        self.stats["read_bytes"] += sizes[key]
                    except Exception as exc:
                        error = error or exc
                if error is not None:
                    raise error
            finally:
                self.stats["read_seconds"] += time.monotonic() - start
            self.stats["hits"] += len(keys) - len(missing)
            self.stats["misses"] += len(missing)
            for key in keys:
                self._entries.move_to_end(key)
            yield {key: self._entries[key][0] for key in keys}
        finally:
            self._leased = False

    def close(self):
        self._pool.shutdown(wait=True)
        self._entries.clear()
        self.bytes_used = 0

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
