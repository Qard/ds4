# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface-hub==2.2.0"]
# ///
"""Download the pinned MLX Muse checkpoint, including tokenizer and all shards."""

import argparse
import shlex
from pathlib import Path

from huggingface_hub import snapshot_download


REPO = "mlx-community/Muse-Spark-Open-816B-A42B-oQ3e"
REVISION = "b4ce3d89d009b5a76ee8466717f708b327e3de75"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "models" / "Muse-Spark-Open-816B-A42B-oQ3e"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true", help="Fetch metadata only; do not download weights")
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    print(f"Model: {REPO}\nRevision: {REVISION}\nDestination: {output}", flush=True)
    result = snapshot_download(REPO, revision=REVISION, local_dir=output,
                               dry_run=args.dry_run, max_workers=2)
    if args.dry_run:
        total = sum(item.file_size or 0 for item in result)
        missing = sum(item.file_size or 0 for item in result if item.will_download)
        print(f"Complete checkpoint: {total / 1024**3:.2f} GiB")
        print(f"Files needing download: {missing / 1024**3:.2f} GiB")
        print("Allow roughly 400 GiB free for a fresh download. Partial files may reduce actual transfer.")
    else:
        print("Download complete. Rerunning this command reuses completed files and resumes interrupted transfers.")
        print(f"Inspect: ./ds4-muse -m {shlex.quote(str(output))} --inspect")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
