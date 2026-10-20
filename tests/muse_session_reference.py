# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2,<3"]
# ///
import subprocess
import sys
import tempfile
from pathlib import Path
from muse_graph_reference import checkpoint
with tempfile.TemporaryDirectory() as tmp:
    path=Path(tmp)/'tiny.gguf';checkpoint(path)
    for streamed in (0,1):
        subprocess.run([sys.argv[1],str(path),str(streamed)],check=True)
