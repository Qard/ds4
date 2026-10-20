"""Opt-in real-weight parity probe (~21 MiB download, no full model required).

Run from the repository root with the Muse Python environment:
    PYTHONPATH=. python tests/probe_muse_expert.py --output /tmp/muse-expert-probe

This file is not an automatic test because it uses the network.
The saved fixture contains only one expert and
cannot be used as a complete model checkpoint.
"""

import argparse
import json
import struct
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.muse_spark import PreciseSwiGLU
from mlx_lm.models.switch_layers import SwitchGLU

from muse.runtime import StreamedExperts, quant_params, read_array
from muse.storage import ExpertCache, TensorStore


REPO = "mlx-community/Muse-Spark-Open-816B-A42B-oQ3e"
REVISION = "b4ce3d89d009b5a76ee8466717f708b327e3de75"


def fetch_json(filename):
    with urllib.request.urlopen(f"https://huggingface.co/{REPO}/resolve/{REVISION}/{filename}", timeout=60) as r:
        return json.load(r)


def fetch_range(filename, start, length):
    end = start + length - 1
    url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{filename}?slice={start}-{end}"
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        content_range = r.headers.get("Content-Range", "")
        if r.status != 206 or not content_range.startswith(f"bytes {start}-{end}/"):
            raise OSError(f"Server did not honor exact byte range: {r.status} {content_range}")
        data = r.read(length + 1)
        if len(data) != length:
            raise OSError("Truncated or oversized range response")
        return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--expert", type=int, default=0)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = fetch_json("config.json")
    weight_map = fetch_json("model.safetensors.index.json")["weight_map"]
    if not 0 <= args.expert < config["text_config"]["num_local_experts"]:
        raise ValueError("Invalid expert index")
    prefix = f"model.language_model.layers.{args.layer}.mlp.experts."
    names = [prefix + proj + "." + field for proj in ("gate_proj", "up_proj", "down_proj")
             for field in ("weight", "scales", "biases")]
    headers = {}
    for shard in sorted({weight_map[name] for name in names}):
        length = struct.unpack("<Q", fetch_range(shard, 0, 8))[0]
        if length > 32 * 1024**2:
            raise ValueError("Oversized safetensors header")
        headers[shard] = (8 + length, json.loads(fetch_range(shard, 8, length)))

    def download(name):
        shard = weight_map[name]
        base, header = headers[shard]
        entry = header[name]
        lo, hi = entry["data_offsets"]
        row_bytes = (hi - lo) // entry["shape"][0]
        data = fetch_range(shard, base + lo + args.expert * row_bytes, row_bytes)
        return name, {**entry, "shape": [1, *entry["shape"][1:]]}, data

    fixture = args.output / "model.safetensors"
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(download, names))
    header, payload = {}, bytearray()
    for name, entry, data in rows:
        lo = len(payload)
        payload.extend(data)
        header[name] = {**entry, "data_offsets": [lo, len(payload)]}
    encoded = json.dumps(header).encode()
    fixture.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
    del rows, payload

    with TensorStore(args.output) as store:
        def load(key):
            return {proj: {field: read_array(store, prefix + proj + "." + field, row=0)
                           for field in ("weight", "scales", "biases")}
                    for proj in ("gate_proj", "up_proj", "down_proj")}

        width = config["text_config"]["moe_hidden_size"]
        intermediate = config["text_config"]["moe_intermediate_size"]
        reference = SwitchGLU(width, intermediate, 1, activation=PreciseSwiGLU())
        nn.quantize(reference, group_size=64, bits=3,
                    class_predicate=lambda path, mod: quant_params(config, prefix + path)
                    if hasattr(mod, "to_quantized") else False)
        reference.load_weights([(name[len(prefix):], read_array(store, name)) for name in names])
        mx.eval(reference.parameters())
        with ExpertCache(store.expert_bytes(args.layer), lambda _: store.expert_bytes(args.layer), load) as cache:
            streamed = StreamedExperts(args.layer, nn.RMSNorm(width), cache, config, store)
            mx.random.seed(42)
            inputs = mx.random.normal((1, 3, width)).astype(mx.bfloat16)
            indices = mx.zeros((1, 3, 1), dtype=mx.uint32)
            expected = reference(inputs, indices)
            mx.eval(expected)
            started = time.monotonic()
            actual = streamed(inputs, indices)
            mx.eval(actual)
            elapsed = time.monotonic() - started
            diff = expected.astype(mx.float32) - actual.astype(mx.float32)
            relative_rms = (mx.sqrt(mx.mean(diff * diff)) /
                            mx.maximum(mx.sqrt(mx.mean(expected.astype(mx.float32)**2)), 1e-12)).item()
            report = dict(repo=REPO, revision=REVISION, layer=args.layer, expert=args.expert,
                          expert_bytes=store.expert_bytes(args.layer), max_abs_error=mx.max(mx.abs(diff)).item(),
                          relative_rms_error=relative_rms, streamed_seconds=elapsed,
                          peak_mlx_bytes=mx.get_peak_memory(), cache_stats=cache.stats.copy())
            print(json.dumps(report, indent=2))
            (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            if relative_rms >= 0.02:
                raise AssertionError(f"Real-expert parity failed: {relative_rms}")


if __name__ == "__main__":
    main()
