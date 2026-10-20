import contextlib
import io
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from muse.download import main, REPO, REVISION


class DownloadTests(unittest.TestCase):
    def test_dry_run_reports_only_missing_files_without_fetching_weights(self):
        calls = []
        def snapshot(repo, **kwargs):
            calls.append((repo, kwargs))
            self.assertTrue(kwargs["dry_run"], "dry-run attempted a payload download")
            return [SimpleNamespace(file_size=1024**3, will_download=False),
                    SimpleNamespace(file_size=2 * 1024**3, will_download=True)]
        with tempfile.TemporaryDirectory() as tmp, patch("muse.download.snapshot_download", snapshot):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(main(["--output", tmp, "--dry-run"]), 0)
            self.assertIn("3.00 GiB", out.getvalue())
            self.assertIn("2.00 GiB", out.getvalue())
            self.assertEqual(calls[0][0], REPO)
            self.assertEqual(calls[0][1]["revision"], REVISION)

    def test_existing_shell_downloader_routes_muse_arguments_without_splitting(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            uv = Path(tmp) / "uv"
            uv.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
            uv.chmod(0o755)
            result = subprocess.run(
                ["sh", str(root / "download_model.sh"), "muse-oq3e",
                 "--output", "/Volumes/Fast SSD/Muse", "--dry-run"],
                env={**os.environ, "PATH": tmp + os.pathsep + os.environ["PATH"]},
                capture_output=True, text=True, check=True)
            self.assertEqual(result.stdout.splitlines(), [
                "run", "--script", str(root / "muse/download.py"),
                "--output", "/Volumes/Fast SSD/Muse", "--dry-run"])


if __name__ == "__main__":
    unittest.main()
