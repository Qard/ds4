# Native Muse Spark Open integration

[README](../README.md)

Muse runs through the native `ds4` C/Metal engine and its bounded SSD expert
cache. There is no separate Muse runner or MLX inference dependency. Native
loading, tokenizer, graph, sessions, KV persistence, CLI, server, evaluation, and agent
adapters are implemented. Small independent fixtures exercise numerical
correctness and the conversation protocol.

**Real-checkpoint startup, full-vocabulary prefill consistency, KV round trips,
live APIs, an 8K context sweep and the no-reasoning agent compaction/recovery
suite pass. Final cross-chunk and reasoning-mode validation are in progress.
This branch is not yet production-ready.**
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
  --ctx 4096 --nothink --tokens 256 -p 'Hello'
```

Muse selects SSD expert streaming automatically. The standard `ds4-server` and
`ds4-agent` model options select the same engine. Chat uses Muse recipient
headers for reasoning and ATEM tool calls, including consecutive tool messages.
The OpenAI, Anthropic, and Responses adapters separate reasoning from visible
content. Text containing literal `<think>` tags stays ordinary content.

Only bodies addressed to a tool enter the executable ATEM parser. ATEM examples
inside user-facing or commentary messages remain literal text. The server streams
public text normally and validates the complete set of same-turn calls before
emitting their API tool events at the native turn boundary.

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
uses the bounded batch loader. Single-token execution also advances ds4's
existing frequency-cache aging clock once per decoded token, so accumulated
prefill popularity decays as the conversation moves to new topics.

Admission reserves fixed weights, per-session KV and scratch, a safety margin,
and the expert cache. It also measures currently available host memory before
pinning weights. Another application's GPU allocations therefore reduce the
allowed budget. Multiple sessions reserve independent KV state. Explicit cache
requests are capped to the safe budget and must fit at least eight experts. If a
frontend chooses a larger context after tokenization, the first session can
refit the cache before allocating its graph. Live sessions never resize that
cache underneath another session.

Completed full-checkpoint context sweeps are reported below. Muse defaults to
512-token prefill chunks; explicit `--prefill-chunk` settings override this.
Both graph admission and session allocation use that same default.

## Checkpoint-free verification

```sh
make test-muse-gguf test-muse-tokenizer
make test-muse-chat test-muse-cli test-muse-server test-muse-agent test-muse-eval test-muse-engine
make test-muse-metal test-muse-attention test-muse-experts test-muse-quant
make test-muse-graph test-muse-session test-muse-memory
uv run --with huggingface-hub==2.2.0 --with 'numpy>=2,<3' --with gguf==0.19.0 \
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
expert caches, prefill chunks 32/128/512, teacher-forced continuations, and full-logit
comparisons with `tests/muse_compare_frontiers.py`. API checks cover OpenAI,
Anthropic and Responses text/streaming output, literal `<think>` text, reasoning
channels, and named tool calls/results. The lookup tool returns a random token
without executing code. A separate probe requests two calls before waiting,
returns a different random token for each call ID, and checks their ordered final
answer in JSON and SSE for all three APIs:

```sh
python3 tests/muse_live_tools.py --output /tmp/muse-multiple-tools --plan
python3 tests/muse_live_tools.py --output /tmp/muse-multiple-tools
```

A separate run after restarting the server requires a
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

## Independent checkpoint arithmetic

`tests/muse_checkpoint_reference.py` computes one short prefix with NumPy and
llama.cpp's Python GGUF quant decoder. It is a numerical diagnostic, with no
sampling, conversation handling or production runner. It uses bounded reads of
256 matrix rows and only the routed experts, without mapping the checkpoint or
allocating a model-sized CPU cache. Prefixes are limited to 32 tokens. The small
packed graph independently verifies this diagnostic before full-checkpoint use.

Use prompt IDs saved by the native diagnostic, retaining a new output filename:

```sh
VECLIB_MAXIMUM_THREADS=4 uv run --script tests/muse_checkpoint_reference.py \
  --model "$HOME/.ds4/models/muse-spark-open-native.gguf" \
  --prompt-ids /tmp/muse-native-validation/prefill-32/raw.prompt.i32 \
  --output /tmp/muse-raw-reference.f32
```

The output is all 202,048 FP32 logits at the final prompt token. Compare it with
the first vocabulary vector in the native `raw.logits.f32`, rather than a vector
after generated continuation tokens. Close resident model processes for an
uncontended comparison. This calculation is intentionally slow and independent
of the native Metal execution path.

The corrected native chat framing was revalidated on the full checkpoint at
prefill chunks 1 and 32: all vocabulary logits, snapshots, paired decoding and
rewinds again matched exactly. Live OpenAI Chat Completions, Anthropic Messages
and Responses checks passed plain text, streaming, literal tag preservation and
tool round trips. Streamed tool names and arguments passed in all three formats.
The first request after restarting the server restored all 155 prompt tokens
from disk. The server held about 84 GiB resident with its automatic expert cache.

The real agent processed its 1341-token system prompt, interrupted generation in
0.92 seconds, and then used `bash` to create and read the requested test file.
Its recovery prefills reused 1449 and 1604 tokens. The CLI also returned only the
requested `CLI_OK` with a 512-token prefill setting and private output disabled.
A sampled high-effort arithmetic request returned `323` with its reasoning on a
separate stream. The earlier strict-code-phrase high-effort sample produced an
incorrect refusal; it remains a failed quality probe.

The independent four-token checkpoint calculation agreed on the top ten tokens
and had logit cosine similarity 0.9999903, maximum error 0.087874 and RMS error
0.017334. Layer traces locate the first routing difference at layer 45: the
native eighth/ninth scores are separated by 4.1e-7. Residual maximum error was
0.000512 at layer 44 and jumped to 0.029175 after that selection. NumPy selected
the same experts as Metal when given Metal's router logits. A controlled tail
calculation starting from the independent layer-44 residual and holding native
expert choices fixed reduced final maximum/RMS logit error to 0.002533/0.000566
and increased cosine similarity to 0.999999991. This isolates a near-tie routing
change under different FP32 reductions and FP16 KV rounding; it is not an exact
independent golden-logit pass. Native chunk/session consistency remains exact.

Build `tests/test_muse_checkpoint_trace` to repeat the per-layer diagnostic with
`MODEL PROMPT_IDS NEW_OUTPUT_DIRECTORY`. Its observer is compiled only into the
test binary. It saves residuals, normalized FFN inputs, router logits, selected
experts, queries and FP16 keys. Adding the observer preserved all native final
logits exactly. `muse_checkpoint_reference.py --trace-directory NEW_DIRECTORY`
saves the corresponding independent values.


## Sustained automatic-cache run

A sequential teacher-forced benchmark completed the 1024, 2048, 4096 and 8192
context frontiers in 1556 seconds with prefill chunks of 512. It used 5336 cached
experts (64.49 GiB), with an 84.83 GiB allocation plan. Each frontier decoded 64
tokens; reported prefill rates apply to the additional tokens at that frontier.

| Context | Additional prefill tokens | Prefill tokens/s | Decode tokens/s |
| --- | --- | --- | --- |
| 1024 | 1024 | 8.45 | 5.29 |
| 2048 | 1024 | 5.31 | 2.93 |
| 4096 | 2048 | 5.16 | 3.35 |
| 8192 | 4096 | 5.30 | 3.91 |

All frontiers completed, including snapshots restored between frontiers and
execution beyond the 2048-token sliding window. At the 1024 frontier, all 202,048
logits matched exactly between prefill chunks 32, 512 and 2048. The 8192 frontier
also matched exactly against a fresh 128-token-chunk run, across all 202,048
logits, with the cache-aging update enabled.

The memory collector retained 35 samples over approximately 17 minutes of this
run, with a sampled peak resident size of 84.42 GiB and no increase in the system
Swapouts counter during that sampled interval. This does not mean the machine
had no preexisting swap use or that the collector covered the entire process.
The completed sweep preceded the frequency-cache aging fix; that policy update
passed its full-checkpoint rerun: 8192 prompt tokens plus 64 continuation tokens
completed in 1543 seconds, with 5.44 prefill tokens/s and 3.27 decode tokens/s.
The complete 78-sample record peaked at 83.49 GiB process RSS, with no increase
in Swapouts. A mid-run VM map showed a read-only mapped-weight region resident at 18.3 GiB
and a 66.6 GiB physical footprint; GPU alias mappings
are not additional unique allocations.


The recipient-separation build passed a fresh 27-request API run across all three
formats, including literal ATEM documentation with tools enabled in both JSON
and SSE. It returned no tool calls for those examples, suppressed private text
at disabled effort, and returned `323` with a separate reasoning stream at high
effort. A further 12 requests passed two same-turn calls followed by two distinct
random results in JSON and SSE for every API. Both IDs, lookup keys, and the
ordered final answer were checked. The reusable `muse_live_tools.py` parser also
reconstructed exactly the calls saved by all three actual SSE runs.

Grouped raw ATEM replay now consumes a complete block per group of invokes,
using the same grammar as generation parsing. Regressions cover two invokes in
one block, separate blocks, and closing-looking block text inside a parameter,
without duplicating calls or splitting parameter content.


## Full agent compaction and recovery baseline

The 8192-context, 512-token-prefill agent run passed its complete no-reasoning
suite in 5530 seconds. The monitor retained 277 samples, with a peak process RSS
of 83.28 GiB and no increase in system Swapouts over that sampled run.

- Normal compaction reduced 7101 tokens to 2317 while retaining an 819-token
  verbatim tail. The agent then created and read the requested `COMPACTION_OK` file.
- A saved 2798-token session restored in about 0.7 seconds. Its first tool prompt
  reused all 2798 tokens, and the agent created and read `AFTER_RESTORE`.
- Context exhaustion at 8040 tokens summarized a bounded 6971-token prefix and
  retained all 1069 unsummarized tokens. The rebuilt context had 2583 tokens;
  the agent then created and read the requested `RECOVERED` file.
- Code generation reached a context boundary after 416 tokens, compacted, and
  resumed inside the same assistant message. The result contained exactly 80
  complete functions in ascending order, compiled, and passed an executable
  oracle checking every function's squared result.
- Oversized input was rejected, and the following turn successfully created
  `AFTER_REJECTION` while reusing the prior valid conversation.

The first two compaction exchanges took about 25.7 and 26.7 minutes before
resumed generation. These cases deliberately submit a large new archive, so most of the summary
source is not already in live KV. The times include summary prefill, summary
generation and context rebuilding; they are practical cold-input latency
measurements, not isolated kernel throughput. This baseline preceded the system-prefix reuse optimization.
The focused follow-up passed; the full reasoning-mode run remains in progress.

The Q8 dispatch experiment compared an 8192-by-8192 projection over 512 input
rows. All 4,194,304 output floats were bit-identical, but a wider dispatch took
19.36 ms versus 13.05 ms for the existing eight-row slices, confirmed by a second
baseline run. The wider variant was discarded without changing the production
path.


The focused 4096-context check passed with system-prefix reuse enabled. The
existing checkpoint restored all 1341 system tokens in 47.9 ms; the subsequent
1978-token rebuild reported `cached=1341 suffix=637` and completed that suffix
in 119 seconds. The resumed agent created and read `COMPACTION_OK`. The check
requires a complete successful cache ledger, so a later two-token assistant
sync cannot hide a cold compaction rebuild. Its 47 memory samples peaked at
83.13 GiB process RSS with no increase in Swapouts.

This uses the existing agent KV loader, including its model, quantization and
exact-text checks, and additionally verifies the complete token prefix before
reporting a hit. It does not replace the conversation transcript. Missing or
invalid cache files fall back to the ordinary sync. The native loader, graph
and arithmetic are unchanged by this optimization.

Parallel tool results are rendered in the preceding assistant call order, even
when replies arrive in reverse order. Output-only Responses and Anthropic
continuations retain owned ID/name bindings at the existing live KV frontier;
reused IDs in replayed history resolve to the nearest preceding assistant call.
These rules have checkpoint-free regressions, including both live API styles.
