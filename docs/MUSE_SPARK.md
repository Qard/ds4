# Experimental Muse Spark Open on a 128 GB Mac

[README](../README.md)

This branch adds `ds4-muse`, a text-only runner with explicit SSD expert reads
and a byte-bounded cache. It uses MLX-LM's existing Muse graph for attention,
routing, normalization, residuals, and KV state, replacing only expert execution
and weight loading.

**Status: small-model inference and a real 816B-checkpoint expert are tested on
an M5 Max. Full 816B generation and useful throughput are not verified.**
The test machine had 128 GiB RAM and approximately 331 GiB free storage; the
selected checkpoint needs approximately 346 GiB. No full checkpoint was downloaded.

This is an experimental entry point inside the fork, not a native C/Metal port.
`ds4`, `ds4-server`, and `ds4-agent` retain their existing model support. There is
no Muse server, integrated tool execution, vision, audio, or speculative decoding
yet. The CLI uses Muse's chat template and emits its text stream, including any
reasoning/protocol markers the tokenizer preserves.

## Running

Use a **converted MLX affine** checkpoint such as
[`mlx-community/Muse-Spark-Open-816B-A42B-oQ3e`](https://huggingface.co/mlx-community/Muse-Spark-Open-816B-A42B-oQ3e).
The inspected revision is `b4ce3d89d009b5a76ee8466717f708b327e3de75`.
All shards, the index, config, and tokenizer files must be present locally.
Nothing is downloaded by the runner. Model weights remain subject to the
Muse Model License.

Download the audited revision with the existing download entry point. Install
uv first, then choose a fast SSD with roughly 400 GiB free for a fresh download:

```sh
./download_model.sh muse-oq3e --output /Volumes/FastSSD/Muse --dry-run
./download_model.sh muse-oq3e --output /Volumes/FastSSD/Muse
./ds4-muse -m /Volumes/FastSSD/Muse --inspect
./ds4-muse -m /Volumes/FastSSD/Muse --ctx 8192 --tokens 32 -p 'Hello'
```

Without `--output`, the downloader uses `models/Muse-Spark-Open-816B-A42B-oQ3e`
inside the checkout. Rerun the same command to resume interrupted downloads;
completed files are reused. It downloads all shards and supporting files directly
into the destination through Hugging Face Hub's `local_dir` mechanism. `HF_TOKEN`
is honored if authentication or higher download limits are needed. `--dry-run`
requests metadata only and reports total size and files still needing download.

Install [uv](https://docs.astral.sh/uv/) and run the executable from the checkout.
Its inline dependencies and adjacent lockfile pin MLX-LM 0.32.2 and MLX 0.32.4.
uv creates an isolated environment; the native build does not depend on MLX.

```sh
./ds4-muse --help
./ds4-muse -m /path/to/local/Muse-Spark-Open-816B-A42B-oQ3e --inspect
./ds4-muse -m /path/to/local/Muse-Spark-Open-816B-A42B-oQ3e \
  --ctx 8192 --tokens 32 --prefill-chunk 16 \
  --reasoning-effort low -p 'Write a Python function that reverses a linked list.'
```

Use the automatic budget first. `--cache-gib 48` requests a 48 GiB expert cache;
it must fit *after* resident weights, planned KV state, and scratch. `--memory-gib`
changes the total planned working set. Increasing it does not help an already
memory-pressured machine. Greedy sampling is the validation default; `--temp`
enables sampling. Prompt plus maximum generation must fit `--ctx`.

Per-projection mixed affine quantization is supported, retaining packed weights,
scales, and quantization biases together. Raw NVFP4/BF16 checkpoints, unsupported
layouts, and GGUFs are rejected. Only oQ3e's actual layout has been audited;
other quantizations need their own correctness checks.

## Memory and execution

On the 128 GiB M5 Max, the audited oQ3e headers at 32K context produce:

| Component | Planned GiB |
| --- | ---: |
| Resident text weights | 16.881 |
| BF16 KV state | 2.359 |
| Scratch and allocator headroom | 4.000 |
| Dynamic expert cache | 62.776 |
| Total planned working set | 86.016 |

The total defaults to 80% of Metal's recommended working set, which is smaller
than physical RAM. Vision/audio weights are excluded. KV accounting includes
256-token capacity increments. Scratch covers temporaries, allocation rounding,
and sliding-window prefill overhead; it is a reserve, not an RSS guarantee.
The CLI limits MLX's freed-buffer cache to 256 MiB. MLX memory limits are
allocator controls, not OS hard limits.

Headers, shapes, and dtypes are validated before mandatory text weights are
materialized. Random expert banks are removed before model evaluation. The full
model skeleton used only 7,492 active MLX bytes during the header audit.

The LRU cache spans all layers. Selected expert rows are read with `pread` on a
small thread pool. A lease admits only experts that fit, evicting old entries
before allocating replacements. GPU work using leased arrays is evaluated before
eviction can occur. Cache arrays live outside the model parameter tree, so
whole-model `mx.eval` cannot accidentally materialize expert banks.

Wide prefills are split into expert groups when needed; even a one-expert cache
works without staging full layers or discarding selected experts. Separate
quantized matmuls avoid copying cached experts into a stacked bank. This first
implementation synchronizes at routing/group boundaries. It has parallel reads
but no disk/GPU overlap or predictive prefetch.

Statistics report hits, misses, evictions, bytes read, admission wall time,
stage times, and peak active MLX allocation. Admission time includes array
creation. OS page-cache hits can accelerate repeated probes without indicating
physical SSD bandwidth.

## Verification

For tests outside uv's script environment:

```sh
uv venv --python 3.13 .venv-muse
uv pip install --python .venv-muse/bin/python \
  'mlx-lm==0.32.2' 'mlx==0.32.4' 'numpy>=2,<3'
.venv-muse/bin/python -m unittest discover -s tests -p 'test_muse_*.py' -v
```

The 22 tests cover downloader dry-run/argument routing, cross-shard slicing,
truncated files, cache admission/eviction,
failed reads, mixed 3/4/5-bit projections, single-expert cache prefill, reference
logits through sliding-window rollover and decode, memory refusal, malformed
checkpoints, checkpoint chat templates, and numeric CLI generation. Missing resident tensors cause an error
instead of leaving random initialized weights.

The small two-layer Muse fixture matched reference logits exactly over the
tested prefill/decode sequence with forced cache eviction. A full checkpoint
header audit checked 558 expert tensors and 1,431 resident text tensors against
expected names and shapes with no mismatches.

An opt-in probe downloads only one expert from the pinned checkpoint:

```sh
PYTHONPATH=. .venv-muse/bin/python tests/probe_muse_expert.py \
  --output /tmp/muse-expert-probe
```

It saves a 21 MiB expert-only fixture and a JSON report. That directory is not
a complete model. Against MLX's unmodified `SwitchGLU` on the M5 Max, layer 0,
expert 0 gave maximum absolute error `0.0001220703125`, relative RMS error
`0.0000206311`, and approximately 42.3 MiB peak active MLX allocation. This
validates real-weight expert execution, not model quality or generation speed.

Existing native checks also passed:

```sh
make -j6
make -j6 ds4_test test-ssd-cache
./ds4_test --server
make cpu -j6
```

The full native model correctness/speed suite was not run because no supported
native GGUF was loaded. No native inference source was changed.

## Next milestones

1. Put the complete checkpoint on a sufficiently large fast local SSD and run
   short generation. Compare logits/greedy tokens with a resident reference on
   a machine that fits the same quantization.
2. Measure cold/warm initial prefill, short tool-result appends, generation rate,
   memory pressure, expert hit rate, and SSD bytes per generated token on real
   coding prompts.
3. Optimize the measured bottleneck: Muse-specific routing locality, read/compute
   overlap, and gathered expert dispatch without duplicating cached weights.
4. Decide whether to retain this MLX path or port Muse into the native graph and
   server. The native SSD cache is a design reference; this prototype does not
   call its C API.

Eight experts across 62 layers, at 21 MiB per expert, can require approximately
10.2 GiB of cold reads per generated token. Cache locality determines whether
this becomes practical on one laptop. Memory fit does not establish speed.
