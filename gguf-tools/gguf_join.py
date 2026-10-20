"""Join split GGUFs without requantization or a second full-model staging copy.

Headers can be small range-fetched files when sizes and acquire/release hooks
are supplied. This lets a downloader stage one shard, append it, commit its
journal, and release that shard before downloading the next one.
"""

import fcntl
import hashlib
import json
import math
import os
import struct
from pathlib import Path

from qwen4_pack_to_qwen4exp import Reader, kv_bytes, w_str


LAYOUT = {0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20),
          8: (32, 34), 10: (256, 84), 11: (256, 110), 12: (256, 144),
          13: (256, 176), 14: (256, 210), 16: (256, 66),
          26: (1, 4), 27: (1, 8), 30: (1, 2), 39: (32, 17)}


def tensor_bytes(kind, dims):
    if kind not in LAYOUT or not dims or any(type(d) is not int or d <= 0 for d in dims):
        raise ValueError(f'Unsupported tensor kind/shape: {kind} {dims}')
    block, size = LAYOUT[kind]
    if dims[0] % block:
        raise ValueError('Tensor row is not block-aligned')
    return math.prod(dims) // block * size


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def save_journal(path, state):
    update = Path(str(path) + '.tmp')
    with update.open('w') as fp:
        json.dump(state, fp)
        fp.flush()
        os.fsync(fp.fileno())
    os.replace(update, path)
    sync_directory(path.parent)


def join_gguf(headers, output, *, sizes=None, acquire=None, release=None):
    headers, output = [Path(p) for p in headers], Path(output)
    if not headers:
        raise ValueError('No input GGUF shards')
    sizes = sizes or [p.stat().st_size for p in headers]
    if len(sizes) != len(headers):
        raise ValueError('Shard size count mismatch')
    readers = []
    try:
        for path in headers:
            readers.append(Reader(str(path)))
        metadata = [(k, readers[0].kv_types[k], v) for k, v in readers[0].kv.items()
                    if not k.startswith('split.')]
        records, seen, offset = [], set(), 0
        for index, reader in enumerate(readers):
            if reader.version != 3 or reader.kv.get('split.no') != index or reader.kv.get('split.count') != len(headers):
                raise ValueError('Invalid split GGUF numbering/version')
            # Secondary shards can omit model metadata, but must not contradict
            # metadata they do carry. Tokenizer arrays need structural comparison.
            for key, kind, value in metadata:
                if key in reader.kv:
                    a, b = getattr(value, 'values', value), getattr(reader.kv[key], 'values', reader.kv[key])
                    if kind != reader.kv_types[key] or a != b:
                        raise ValueError(f'Inconsistent shard metadata: {key}')
            previous_end = reader.data_start
            for name, (kind, dims, source_offset) in sorted(reader.tensors.items(), key=lambda pair: pair[1][2]):
                if name in seen:
                    raise ValueError(f'Duplicate tensor: {name}')
                seen.add(name)
                nbytes = tensor_bytes(kind, dims)
                start = reader.data_start + source_offset
                if start < previous_end or start + nbytes > sizes[index]:
                    raise ValueError(f'Overlapping or truncated source tensor: {name}')
                previous_end = start + nbytes
                offset = (offset + 31) // 32 * 32
                records.append((index, name, kind, dims, start, offset, nbytes))
                offset += nbytes
        expected_count = readers[0].kv.get('split.tensors.count')
        if expected_count != len(records):
            raise ValueError(f'Split tensor count mismatch: {expected_count} != {len(records)}')
        header = b'GGUF' + struct.pack('<IQQ', 3, len(records), len(metadata))
        header += b''.join(kv_bytes(*entry) for entry in metadata)
        for index, name, kind, dims, start, target, nbytes in records:
            header += w_str(name) + struct.pack('<I', len(dims))
            header += struct.pack('<' + 'Q' * len(dims), *dims) + struct.pack('<IQ', kind, target)
        alignment = readers[0].kv.get('general.alignment', 32)
        if alignment != 32:
            raise ValueError('Joining currently requires 32-byte GGUF alignment')
        header += b'\0' * (-len(header) % alignment)
        fingerprint = hashlib.sha256(header + json.dumps(sizes).encode()).hexdigest()
        for reader in readers:
            reader.f.close()
        readers.clear()

        output.parent.mkdir(parents=True, exist_ok=True)
        pending = Path(str(output) + '.incomplete')
        journal = Path(str(output) + '.progress.json')
        # Keep the lock inode: unlinking it permits concurrent unrelated locks.
        with Path(str(output) + '.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if output.exists():
                raise FileExistsError(output)
            if pending.exists() != journal.exists():
                raise ValueError('Incomplete GGUF and progress journal must both exist; preserve them for diagnosis')
            if pending.exists():
                state = json.loads(journal.read_text())
                if state.get('fingerprint') != fingerprint:
                    raise ValueError('Resume journal does not match the pinned GGUF layout')
                with pending.open('rb') as check:
                    if check.read(len(header)) != header:
                        raise ValueError('Resume GGUF header differs from the pinned layout')
                if not 0 <= state['next_shard'] <= len(headers) or not len(header) <= state['bytes'] <= len(header) + offset:
                    raise ValueError('Invalid GGUF progress journal')
                committed = [r for r in records if r[0] < state['next_shard']]
                boundary = len(header) + (committed[-1][5] + committed[-1][6] if committed else 0)
                if state['bytes'] != boundary:
                    raise ValueError('Invalid GGUF progress journal boundary')
                if pending.stat().st_size < state['bytes']:
                    raise ValueError('Truncated resumed GGUF')
            else:
                with pending.open('xb') as fp:
                    fp.write(header)
                    fp.flush()
                    os.fsync(fp.fileno())
                state = dict(fingerprint=fingerprint, next_shard=0, bytes=len(header))
                save_journal(journal, state)

            with pending.open('r+b', buffering=16 * 1024**2) as dst:
                dst.truncate(state['bytes'])
                for index in range(state['next_shard'], len(headers)):
                    entries = [record for record in records if record[0] == index]
                    if entries:
                        source = Path(acquire(index) if acquire else headers[index])
                        if source.resolve() == pending.resolve() or source.stat().st_size != sizes[index]:
                            raise ValueError('Source shard changed size or aliases output')
                        with source.open('rb', buffering=16 * 1024**2) as src:
                            for _, name, kind, dims, start, target, nbytes in entries:
                                dst.seek(len(header) + target)
                                src.seek(start)
                                left = nbytes
                                while left:
                                    chunk = src.read(min(left, 16 * 1024**2))
                                    if not chunk:
                                        raise OSError(f'Short GGUF source read: {name}')
                                    dst.write(chunk)
                                    left -= len(chunk)
                        end = len(header) + entries[-1][5] + entries[-1][6]
                    else:
                        end = state['bytes']
                    dst.flush()
                    os.fsync(dst.fileno())
                    state = dict(fingerprint=fingerprint, next_shard=index + 1, bytes=end)
                    save_journal(journal, state)
                    if entries and release:
                        release(index)
                dst.truncate(len(header) + offset)
                dst.flush()
                os.fsync(dst.fileno())
            os.rename(pending, output)
            sync_directory(output.parent)
            journal.unlink()
            sync_directory(output.parent)
        return output
    finally:
        for reader in readers:
            reader.f.close()
