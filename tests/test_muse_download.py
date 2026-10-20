import contextlib
import hashlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'gguf-tools'))
import download_muse as download


class DownloadTests(unittest.TestCase):
    def inventory(self, root):
        output = root / 'model.gguf'
        staging = root / ('.' + output.name + '.download') / download.REVISION
        staging.mkdir(parents=True)
        files = []
        for index in range(6):
            data = b'metadata' if not index else bytes([index]) * (256 * 1024)
            (staging / f'{index}.header').write_bytes(data)
            files.append(SimpleNamespace(rfilename=download.PREFIX + f'{index + 1:05d}-of-00006.gguf',
                size=len(data), lfs=SimpleNamespace(sha256=hashlib.sha256(data).hexdigest())))
        return output, staging, files

    def api(self, files):
        def info(repo, **kwargs):
            self.assertEqual(repo, download.REPO)
            self.assertEqual(kwargs['revision'], download.REVISION)
            return SimpleNamespace(siblings=files)
        return patch.object(download, 'HfApi', return_value=SimpleNamespace(model_info=info))

    def test_corrupt_cached_metadata_is_rejected_even_for_dry_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            output, staging, files = self.inventory(Path(tmp))
            (staging / '0.header').write_bytes(b'tampered')
            with self.api(files), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
                    download.main(['--output', str(output), '--dry-run'])

    def test_verified_weight_must_match_the_cached_layout_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);output, staging, files = self.inventory(root)
            weight = root / 'weight.gguf';weight.write_bytes((staging / '1.header').read_bytes())
            (staging / '1.header').write_bytes(b'x' * (256 * 1024))
            def join(headers, output, **kwargs):
                kwargs['acquire'](1)
            with self.api(files), patch.object(download, 'hf_hub_download', return_value=str(weight)), \
                    patch.object(download, 'join_gguf', join), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, 'header.*differs'):
                    download.main(['--output', str(output)])

    def test_dry_run_does_not_fetch_weight_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            output, _, files = self.inventory(Path(tmp))
            with self.api(files), patch.object(download, 'hf_hub_download', side_effect=AssertionError('weight fetch')), \
                    patch.object(download, 'join_gguf', side_effect=AssertionError('join')), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(download.main(['--output', str(output), '--dry-run']), 0)

    def test_shell_routes_native_downloader_and_preserves_spaced_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            uv = Path(tmp) / 'uv';uv.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n');uv.chmod(0o755)
            p = subprocess.run(['sh', str(ROOT / 'download_model.sh'), 'muse-gguf',
                '--output', '/Volumes/Fast SSD/Muse.gguf', '--dry-run'],
                env={**os.environ, 'PATH': tmp + os.pathsep + os.environ['PATH']},
                capture_output=True, text=True, check=True)
            self.assertEqual(p.stdout.splitlines(), ['run', '--script', str(ROOT / 'gguf-tools/download_muse.py'),
                '--output', '/Volumes/Fast SSD/Muse.gguf', '--dry-run'])


if __name__ == '__main__':
    unittest.main()
