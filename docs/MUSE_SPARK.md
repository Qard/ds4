# Native Muse Spark Open integration

[README](../README.md)

Muse runs through the native `ds4` C/Metal engine and its bounded SSD expert
cache. There is no separate Muse runner or MLX inference dependency. Native
loading, tokenizer, graph, sessions, KV persistence, CLI, server, evaluation, and agent
adapters are implemented. Small independent fixtures exercise numerical
correctness and the conversation protocol.

**Full-checkpoint generation, numerical parity, sustained memory use, and
throughput remain unvalidated. This branch is not yet production-ready.**
Builds and bounded fixture tests run independently of the downloaded checkpoint.

## Download

```sh
./download_model.sh muse-gguf --dry-run
./download_model.sh muse-gguf
```

The default destination is `~/.ds4/models/muse-spark-open-native.gguf`.
Use `--output /path/to/model.gguf` to change it. Rerun the same command to resume
an interrupted download; a completed output is never overwritten.

The downloader pins revision `c00670568b6ce5e56a247f67091247984df2a01e` of
[`unsloth/Muse-Spark-Open-816B-A42B-GGUF`](https://huggingface.co/unsloth/Muse-Spark-Open-816B-A42B-GGUF),
using the six-part `UD-IQ2_XXS` split. Joining retains the published weight
bytes and rewrites only GGUF metadata and offsets. No requantization occurs.
Model use remains subject to the Muse Model License.

Allow approximately 210 GiB for the joined checkpoint, another 47 GiB for one
staged shard, and free-space headroom. The downloader verifies SHA-256 hashes,
commits each shard to a durable resume journal, and releases its staging copy
before downloading the next shard. `HF_TOKEN` is honored. `uv` installs download
dependencies in an isolated environment; inference requires neither Python nor
MLX. The earlier MLX checkpoint directory uses a different weight format.

## Build and execution path

```sh
make -j6
```

The native command to validate once sufficient memory is available is:

```sh
./ds4 --model ~/.ds4/models/muse-spark-open-native.gguf --metal \
  --ctx 4096 --prefill-chunk 32 --tokens 128 -p 'Hello'
```

Muse selects SSD expert streaming automatically. The standard `ds4-server` and
`ds4-agent` model options select the same engine. Chat uses Muse recipient
headers for reasoning and ATEM tool calls, including consecutive tool messages.
The OpenAI, Anthropic, and Responses adapters separate reasoning from visible
content. Text containing literal `<think>` tags stays ordinary content.

This integration targets text inference on one Apple Silicon GPU. Unsupported
backends, vision, speculative/MTP support models, distributed placement,
steering, and power throttling are rejected before model allocation.

## Memory policy

The downloaded checkpoint contains about 18.25 GiB of fixed weights and 191.81
GiB of expert weights. A cached expert triplet occupies 12.375 MiB. Only the
selected experts are read into explicit cache buffers; routed banks are never
mapped as a cache-overflow fallback.

Admission reserves fixed weights, per-session KV and scratch, a safety margin,
and the expert cache. It also measures currently available host memory before
pinning weights. Another application's GPU allocations therefore reduce the
allowed budget. Multiple sessions reserve independent KV state. Explicit cache
requests are capped to the safe budget and must fit at least eight experts. If a
frontend chooses a larger context after tokenization, the first session can
refit the cache before allocating its graph. Live sessions never resize that
cache underneath another session.

The current kernels and expert scheduling still need full-model performance
validation. Passing fixture tests does not establish useful token throughput.

## Checkpoint-free verification

```sh
make test-muse-gguf test-muse-tokenizer
make test-muse-chat test-muse-server test-muse-agent test-muse-eval test-muse-engine
make test-muse-metal test-muse-attention test-muse-experts test-muse-quant
make test-muse-graph test-muse-session test-muse-memory
uv run --with huggingface-hub==2.2.0 --with 'numpy>=2,<3' \
  python -m unittest discover -s tests -p 'test_muse_*.py'
```

The GGUF admission and memory tests construct sparse headers; they do not load
the full checkpoint. Protocol tests use a metadata-only byte vocabulary and the
published chat template. Metal graph and session tests use small synthetic
weights, independent NumPy reference logits, bounded expert-cache eviction,
sliding-window rollover, rewinds, cancellation, and disk KV round trips.
