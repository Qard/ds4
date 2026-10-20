# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface-hub==2.2.0", "numpy>=2,<3"]
# ///
"""Download and byte-preservingly join the native Muse IQ2_XXS GGUF."""

import argparse
import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, hf_hub_url

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'gguf-tools'))
from gguf_join import join_gguf


REPO = 'unsloth/Muse-Spark-Open-816B-A42B-GGUF'
REVISION = 'c00670568b6ce5e56a247f67091247984df2a01e'
PREFIX = 'UD-IQ2_XXS/Muse-Spark-Open-816B-A42B-UD-IQ2_XXS-'


def verify_sha256(path, expected_hash, filename):
    if not expected_hash:
        raise ValueError(f'Pinned shard has no SHA-256 metadata: {filename}')
    digest = hashlib.sha256()
    with Path(path).open('rb') as fp:
        for chunk in iter(lambda: fp.read(32 * 1024**2), b''):
            digest.update(chunk)
    if digest.hexdigest() != expected_hash:
        raise ValueError(f'Shard SHA-256 mismatch: {filename}')


def header_prefix(filename, path, expected_hash):
    metadata_only = filename.endswith('00001-of-00006.gguf')
    if path.exists():
        if metadata_only:
            verify_sha256(path, expected_hash, filename)
        return
    # The first split holds the tokenizer/metadata and zero weight tensors.
    if metadata_only:
        downloaded = hf_hub_download(REPO, filename, revision=REVISION, local_dir=path.parent / 'metadata')
        verify_sha256(downloaded, expected_hash, filename)
        temp = Path(str(path) + '.tmp')
        shutil.copyfile(downloaded, temp)
        temp.replace(path)
        return
    limit = 256 * 1024
    url = hf_hub_url(REPO, filename, revision=REVISION) + f'?header=0-{limit - 1}'
    request = urllib.request.Request(url, headers={'Range': f'bytes=0-{limit - 1}'})
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.status != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes 0-{limit - 1}/'):
            raise OSError('GGUF header server did not honor the bounded range request')
        data = response.read(limit + 1)
        if len(data) != limit:
            raise OSError('Truncated GGUF header range')
    temp = Path(str(path) + '.tmp')
    temp.write_bytes(data)
    temp.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path.home() / '.ds4/models/muse-spark-open-native.gguf')
    parser.add_argument('--dry-run', action='store_true', help='Fetch small header probes and report size; no full weight shards')
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f'Output already exists; refusing to replace it: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / ('.' + output.name + '.download') / REVISION
    staging.mkdir(parents=True, exist_ok=True)
    info = HfApi().model_info(REPO, revision=REVISION, files_metadata=True)
    files = sorted((f for f in info.siblings if f.rfilename.startswith(PREFIX)), key=lambda f: f.rfilename)
    expected = [PREFIX + f'{i:05d}-of-00006.gguf' for i in range(1, 7)]
    if [f.rfilename for f in files] != expected or any(not f.size for f in files):
        raise ValueError('Pinned GGUF shard inventory differs from the expected six parts')
    headers, sizes = [], []
    for index, file in enumerate(files):
        path = staging / f'{index}.header'
        header_prefix(file.rfilename, path, file.lfs.sha256 if file.lfs else None)
        headers.append(path)
        sizes.append(file.size)
    total, largest = sum(sizes), max(sizes)
    print(f'Checkpoint: {total / 1024**3:.2f} GiB; largest staged shard: {largest / 1024**3:.2f} GiB', flush=True)
    print(f'Output: {output}\nRevision: {REVISION}', flush=True)
    if args.dry_run:
        print('Dry run complete. Only metadata/header probes were fetched.')
        return 0
    pending = Path(str(output) + '.incomplete')
    written = pending.stat().st_size if pending.exists() else 0
    needed = max(0, total - written) + largest + 8 * 1024**3
    if shutil.disk_usage(output.parent).free < needed:
        raise OSError(f'Allow approximately {needed / 1024**3:.1f} GiB free to finish and stage one shard')

    def acquire(index):
        file = files[index]
        path = Path(hf_hub_download(REPO, file.rfilename, revision=REVISION,
                                    local_dir=staging / 'weights'))
        print(f'Verifying shard {index + 1}/{len(files)}', flush=True)
        verify_sha256(path, file.lfs.sha256 if file.lfs else None, file.rfilename)
        # Joining uses cached range probes to calculate source offsets. Verify
        # those bytes against the authenticated shard before copying tensors.
        with path.open('rb') as src, headers[index].open('rb') as header:
            for chunk in iter(lambda: header.read(1024**2), b''):
                if src.read(len(chunk)) != chunk:
                    raise ValueError(f'Cached GGUF header differs from verified shard: {file.rfilename}')
        return path

    def release(index):
        # Only remove files created in this downloader's private staging area,
        # after the joined output and its progress journal are fsynced.
        (staging / 'weights' / files[index].rfilename).unlink()
        print(f'Committed shard {index + 1}/{len(files)}; released staging copy', flush=True)

    join_gguf(headers, output, sizes=sizes, acquire=acquire, release=release)
    print(f'Native checkpoint complete: {output}', flush=True)
    print('Use the native ds4 Metal engine; see docs/MUSE_SPARK.md for validation status.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
