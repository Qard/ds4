# /// script
# requires-python = ">=3.11"
# dependencies = ["tokenizers==0.23.3", "numpy>=2,<3"]
# ///
import subprocess
import sys
import tempfile
from muse_chat_fixture import checkpoint
with tempfile.TemporaryDirectory() as tmp:
    model, _ = checkpoint(tmp)
    subprocess.run([sys.argv[1], str(model)], check=True)
print('Native Muse frontend protocol: PASS')
