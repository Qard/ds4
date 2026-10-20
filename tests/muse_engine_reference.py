# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2,<3"]
# ///
"""Reject unsupported engine modes using sparse headers, without tensor reads."""
import subprocess
import sys
import tempfile
from pathlib import Path
from muse_gguf_fixture import fixture
with tempfile.TemporaryDirectory() as tmp:
    path=Path(tmp)/'sparse.gguf';fixture(path)
    for options in (['--first-token-test'],['--prefill-chunk','8193'],['--ctx','1048577']):
        r=subprocess.run([sys.argv[1],'-m',str(path),'--metal',*options],capture_output=True,text=True)
        assert r.returncode!=0 and 'Muse requires Metal' in r.stderr,(options,r.stderr)
    print('Muse engine option admission: PASS')
