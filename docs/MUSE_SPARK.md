# Native Muse Spark Open integration

[README](../README.md)

Muse runs through the native `ds4` C/Metal engine and its bounded SSD expert
cache. There is no separate Muse runner or MLX inference dependency. Native
loading, tokenizer, graph, sessions, KV persistence, CLI, server, evaluation, and agent
adapters are implemented. Small independent fixtures exercise numerical
correctness and the conversation protocol.

**Real-checkpoint startup, short generation, full-vocabulary prefill consistency,
and KV round trips pass. Long-context, live frontend, sustained memory and
throughput validation are in progress. This branch is not yet production-ready.**
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

The published template always leaves the assistant recipient header open. Muse
has no native zero-reasoning setting: `none` uses its minimum strength (8) and
suppresses private `self` messages in CLI, agent, evaluation and API output.
This preserves the model's ability to select a tool recipient. Private tokens
still consume context and count toward the generation budget. Enabled effort
levels expose their private channel separately from user-visible text.

Muse defaults to temperature 1, top-p 1, no top-k cutoff and min-p 0, matching
[Meta's recommended sampling](https://dev.meta.ai/docs/muse-spark-open/settings).
Explicit client sampling settings take precedence. High effort can require a
large output budget; the live reasoning probe uses 2048 tokens and a fixed
sampling seed. Greedy high-effort decoding exhausted the earlier 512-token
probe while repeating private planning, so that run remains recorded as a
failure rather than a passing reasoning check.

This integration targets text inference on one Apple Silicon GPU. Unsupported
backends, vision, speculative/MTP support models, distributed placement,
steering, and power throttling are rejected before model allocation.

## Memory policy

The downloaded checkpoint contains about 18.25 GiB of fixed weights and 191.81
GiB of expert weights. A cached expert triplet occupies 12.375 MiB. Only the
selected experts are read into explicit cache buffers; routed banks are never
mapped as a cache-overflow fallback.

Single-token decode starts selected-expert reads after routing and overlaps
them with the independent latent projection, using ds4's existing pread pool.
Pending reads are joined before expert dispatch and on graph failure. Prefill
uses the bounded batch loader.

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
make test-muse-chat test-muse-cli test-muse-server test-muse-agent test-muse-eval test-muse-engine
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

## Manual full-checkpoint validation

Build the diagnostic without loading any checkpoint:

```sh
make tests/test_muse_native_live
uv run --script tests/muse_native_validation.py \
  --model "$HOME/.ds4/models/muse-spark-open-native.gguf" \
  --output /tmp/muse-native-validation --plan
```

When enough unified memory is free, remove `--plan` to execute. The output
folder must be new. Runs are sequential, use the native ds4 engine, and request
512 cached experts (about 6.2 GiB), a 4096-token context, and two independently
allocated sessions. Admission still checks currently available memory.

The driver checks raw completion and chat with reasoning disabled and enabled.
It saves every vocabulary logit before and after eight continuation tokens,
checks exact snapshot logits, decodes the same tokens through original and
restored sessions, and checks prefix rewind. The second run teacher-forces the
first run's IDs so prefill chunks 1 and 32 can be compared at identical prefixes.
Prompt IDs, continuation IDs, raw token text, JSON results, and process memory
reports are retained. The fixed continuation length intentionally evaluates
through any stop token; these are numerical checks rather than chat transcripts.

The comparator rejects nonfinite/truncated artifacts, differing prefixes, and
logit errors above 0.002. It reports all-vector maximum/RMS error, cosine
similarity, argmax agreement, and the smallest top-two margin. The snapshot and
paired-session checks use stricter tolerances in the native driver. Tiny Metal
fixtures test the same validation code, and Python tests exercise the artifact
comparator without opening a checkpoint.

These comparisons establish consistency within ds4. They do not replace an
independent full-checkpoint reference, coherent generation and live tool-cycle
checks, or sustained memory and throughput measurements with the automatic
cache budget. Paired-session validation timings include duplicate decoding,
snapshots and rewind work; they are not single-session throughput benchmarks.

## Preparing the complete live test sequence

Preparation does not open a checkpoint or start an inference process:

```sh
python3 tests/muse_prepare_validation.py --output /tmp/muse-live-prepared
```

The new directory contains deterministic benchmark text, its SHA-256, structured
commands, and a runbook. It uses the existing native `ds4-bench`, `ds4-server`,
and `ds4-agent` binaries. Execute its stages sequentially when unified memory
is available. The API client is the only process that runs alongside its server.

The sequence covers context frontiers 1024/2048/4096/8192, bounded and automatic
expert caches, prefill chunks 32/128, teacher-forced continuations, and full-logit
comparisons with `tests/muse_compare_frontiers.py`. API checks cover OpenAI,
Anthropic and Responses text/streaming output, literal `<think>` text, reasoning
channels, and named tool calls/results. The lookup tool returns a random token
without executing code. A separate run after restarting the server requires a
cache hit on its first request, with separate traces retained across restart.
Agent tests cover interruption, actual tools in an isolated test project,
compaction, saved-session restoration, context exhaustion, and recovery. They
run with reasoning disabled and enabled.

A different model is useful for regression testing shared ds4 facilities; its
logits cannot be a numerical reference for Muse. The independent NumPy Muse
reference now covers complete small graphs with F32 weights or Q8 projections
and IQ2_XXS experts, including GQA, sliding-window rollover and expert eviction.
It derives expected values without invoking ds4's kernels or quant decoder.
The packed fixture uses deliberately constructed codebook entries and independent
sign/scale arithmetic; the separate packed-expert tests exercise arbitrary grid
indices. Full-checkpoint checks still compare ds4 execution paths against each
other; independently generated full-checkpoint golden logits are not included.
No additional production runner is required by this test sequence.

## Real-checkpoint validation evidence

On the 128 GiB M5 Max, the pinned native checkpoint produced coherent raw text
and answered the no-reasoning arithmetic prompt with `4`. The fixed-length
numerical driver tested raw completion, chat without reasoning, and chat with
reasoning at prefill chunks 1 and 32. All 202,048 logits matched exactly at the
initial prefix and after each of eight teacher-forced continuation tokens.
Snapshot logits, paired restored-session decoding, and prefix rewinds also
matched exactly. The bounded 1024-expert test requested 12.38 GiB of expert cache
and planned 31.31 GiB including fixed weights and two 4096-token sessions.

These runs exposed a Q8 reduction mismatch between prefill and decode that the
small fixture's numerical tolerance did not catch. Muse now uses the same FP32
row kernel in both paths. The primitive test requires bit-identical results
across batch sizes, and the complete packed graph again requires exact agreement
between its all-row and last-row output heads. Independent NumPy checks still
pass. These results establish short-prefix numerical consistency, rather than
an independent full-checkpoint golden-logit comparison or end-to-end readiness.
