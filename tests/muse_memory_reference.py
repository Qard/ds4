# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2,<3"]
# ///
import subprocess
import sys
import tempfile
from pathlib import Path
from muse_gguf_fixture import fixture
with tempfile.TemporaryDirectory() as tmp:
    path=Path(tmp)/'sparse.gguf';fixture(path)
    subprocess.run([sys.argv[1],str(path)],check=True)
