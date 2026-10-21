# Native Muse Spark Open integration

[README](../README.md)

Muse runs through the native `ds4` C/Metal engine and its bounded SSD expert
cache. There is no separate Muse runner or MLX inference dependency. Native
loading, tokenizer, graph, sessions, KV persistence, CLI, server, evaluation, and agent
adapters are implemented. Small independent fixtures exercise numerical
correctness and the conversation protocol.

**Full-checkpoint numerical comparisons, 8K context tests, live APIs, native
evaluation, and complete agent compaction/recovery suites in both reasoning
modes pass on the 128 GiB M5 Max. The final combined high-reasoning suite also
passes early durable-state retention and later tool use. The default-CLI smoke
check passes. Performance optimization toward 10 decode tokens/s is ongoing.**
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

Run a completion with the standard native binary:

```sh
./ds4 --model ~/.ds4/models/muse-spark-open-native.gguf --metal \
  --ctx 4096 --nothink --tokens 256 -p 'Hello'
```

For the agent, use the same checkpoint and engine:

```sh
./ds4-agent --model ~/.ds4/models/muse-spark-open-native.gguf --metal \
  --ctx 8192 --think --temp 1 --tokens 4096
```

Full-checkpoint validation currently covers contexts through 8192 tokens. Larger
contexts have admission and bounded fixture coverage, but have not completed
full-checkpoint validation on this Mac. Cold compaction of a newly supplied 8K
archive can take tens of minutes; the measured throughput below describes that
practical limitation.

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
make test-muse-metal test-muse-attention test-muse-experts test-muse-quant test-muse-projections
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
python3 tests/muse_live_tools.py --output /tmp/muse-reversed-tools --reverse-results
python3 tests/muse_live_tools.py --output /tmp/muse-live-tools --api anthropic responses --reverse-results --output-only
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
The focused follow-up and full reasoning-mode run passed, including system-prefix reuse.

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

The focused high-reasoning compaction check exposed an empty-summary failure:
Muse spent all 512 internal tokens outside the public summary body. Internal
compaction now supplies `assistant to=user` plus its message boundary and seeds
the existing decoder from that prefix. Both reserve accounting and generation
use the same prefix helper. Ordinary assistant prefixes remain open for native
reasoning and tool selection. The prefix/body regression and real-checkpoint revalidation passed. The fixed
4096-context high-reasoning run completed compaction with a 56-token summary,
restored all 1341 system tokens in 55 ms, rebuilt a 503-token suffix, and created
and read back `COMPACTION_OK` through native tools. The failed run is retained
as evidence.

The two-resident-session server passed six concurrent cross-process tool-result
replays across all three APIs, with each reply pair deliberately reversed.
Both slots restored the 682-token disk prefix. Four fresh JSON/SSE tool-call
cases additionally passed reversed output-only continuations for Anthropic and
Responses: all exact left/right nonce results matched, using the 859-token live
frontier and only 78-89 new result tokens. Disk-cache eviction at its configured
8 GiB limit also completed without disrupting continuations.

The native `ds4-eval` driver passed full-checkpoint smoke cases in both modes:
SuperGPQA Agronomy returned the expected `C` with reasoning disabled (288 prompt
and 203 generated tokens); AIME 2025 number theory returned the expected `70`
with high reasoning (221 prompt and 1193 generated tokens). Only public final
text entered the answer extractor. Both runs exercised the default 512-token
prefill setting. Evaluation retains its shared sampling defaults; the high-mode
case explicitly used temperature 1 with the evaluation default min-p of 0.05.
These two smoke cases are integration evidence, not a benchmark accuracy claim.

## Full high-reasoning agent validation

Before the final compaction fixes described below, the complete 8192-context
high-reasoning suite passed in 5229.6 seconds
(87.2 minutes), using temperature 1, a 4096-token generation budget, and the
default 512-token prefill. Its 262 memory samples peaked at 83.23 GiB process
RSS. The system Swapouts counter increased by 32 pages, or 0.5 MiB; this is a
system-wide counter and does not identify the process responsible.

- Initial compaction reduced 7101 tokens to 2202 while retaining the recent
  819-token tail. Native tools created and read `COMPACTION_OK`.
- Save and restore took 1.33 and 0.72 seconds. The next tool request reused all
  2985 saved tokens and created/read `AFTER_RESTORE`.
- Near-full compaction reduced 8040 tokens to 2455, preserving all 1072
  unsummarized tokens. The agent created and read `RECOVERED`.
- The coding turn hit its boundary after 130 tokens, during private planning.
  Its summary preserved the exact 80-function requirements and identified the
  task as unfinished. Rebuilding restored the 1341-token system prefix and
  evaluated only 411 new tokens. Continuation reused all 1752 rebuilt tokens.
- Public output contained exactly `value_0` through `value_79` in ascending
  order, with every literal square correct. Compilation and the executable
  oracle passed. Generation used 2550 resumed tokens plus 130 carried tokens,
  within the 4096-token budget.
- Oversized input was rejected; the following turn created `AFTER_REJECTION`.

The independent durable-state fixture places a fresh random recovery code
early in the conversation, beyond the retained tail. It requires that exact
code in the actual compaction-summary record, then asks the agent to use the
remembered code in a file without repeating it. Its first live 8K run failed:
the summary acknowledged compaction but omitted the early code. This uncovered
a framing defect: a bounded source cut inside a Muse message was followed by a
new user header without closing the cut message. The adapter now closes that
body before the internal request, keeps incomplete recipient headers entirely
in the retained tail, and reserves space for the possible closing token.
Every token cut through user/assistant headers and bodies has a framing
regression. The framed canary still omitted the code, so valid framing alone
did not resolve semantic retention.

Independent attention fixtures also compare the newest two queries after 8193
positions against an FP64 CPU reference. A global NoPE sentinel requires a key
at position zero to remain visible (a mistakenly clipped window would return
zero); its maximum error was 8.48e-7. The 2048-token sliding-ring/RoPE case had
maximum error 1.02e-6. These checks use small synthetic tensors, not CPU model
inference.

Controlled native prompt probes then restored the same 6967-token source
snapshot before each request;
all 202048 prefix logits restored exactly (maximum error zero). Direct retrieval
returned the exact early code in 21 tokens. The existing summary request returned
only its four-token acknowledgment. Clarifying that the next response must
contain the actual summary text preserved the code in a 73-token summary; an
additional exact-identifiers variant also preserved it (115 tokens). All four
responses stopped normally.

The shared compaction request now uses the tested clarification, replacing the
meta-style acknowledgment-prone preamble. The common fact-preservation rules,
summary budget, and tool restrictions remain the same. The production request
was checked byte-for-byte against the successful controlled input. Final native
validation combines the early-code canary with the complete 8K reasoning-mode
agent recovery suite. On this build, the actual first summary preserved the exact
code from beyond the retained tail, and a later native tool call wrote that code
to a file without the request repeating it. Save/restore also passed, reusing all
3819 saved tokens. Near-limit compaction preserved all 1100 unsummarized tokens;
the subsequent native tools created and read `RECOVERED`. This synthetic
archive-only summary reached its budget by repeating archive text, so that
exchange establishes context recovery rather than useful summarization. The
separate early-code exchange establishes fact retention. Code-continuation and
input-rejection stages follow. The coding turn reached its reserve after 293
tokens and resumed with all 1740 rebuilt tokens cached. Its 1549 resumed tokens
plus 293 carried tokens stayed inside the 4096-token budget. Public output
contained exactly 80 functions in ascending order with every literal square
correct; compilation and the executable oracle passed. This archive-heavy
summary was also uninformative, but the recent task and unfinished reasoning
remained in the verbatim tail. Oversized input was rejected; the subsequent
native tool created `AFTER_REJECTION`, and the turn ended normally.

The complete final combined run on `b380d19` passed in 6052.7 seconds (100.9
minutes). Its 303 memory samples peaked at 82.90 GiB process-tree RSS. The
system-wide Swapouts counter increased by 61788 16-KiB pages (965.4 MiB);
this run was not swap-free. A mid-run VM map reported a 66.3 GiB physical
footprint, 18.3 GiB resident mapped weights, and about 1.2 GiB swapped in a
graphics mapping. GPU aliases make VM region totals larger than unique RAM;
neither those totals nor the system-wide counter identify unique model swap
traffic. Earlier dedicated 8K benchmark and no-reasoning agent runs recorded no
increase in Swapouts under their measured conditions.

The final default-CLI smoke run returned exactly `CLI_OK`, exited successfully,
and confirmed `prefill_cap=512` without an explicit chunk override. It measured
4.32 prefill and 3.01 generation tokens/s on its short 159-token prompt; those
short-run timings are a smoke check, not the sustained 8K benchmark.

## Performance target and comparison

The next target is 10 sustained decode tokens/s at an 8192-token frontier on
the 128 GiB M5 Max, including both private reasoning and public output tokens.
This is an optimization target, not an achieved result.

The [website benchmark table](https://dwarfstar.sh/benchmarks/) mirrors the
[upstream Flash Q2 results](https://github.com/antirez/ds4/blob/main/speed-bench/m5_max.csv).
At 8192 tokens, that baseline reports 683.85 prefill and 37.03 generation
tokens/s. It measures 2048 newly added prompt tokens and a 128-token greedy
probe. Muse's existing 5.44/3.27 result instead measures 8192 prompt tokens and
64 teacher-forced decode tokens, with 128-token prefill chunks. Those are useful
starting observations, but not a matched cross-model benchmark.

[DeepSeek's model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
lists Flash at 284B total / 13B active parameters and Pro at 1.6T / 49B.
[Muse's model card](https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B)
lists 816B / 42B. Muse therefore has about 3.2 times Flash's active parameter
count, and its checkpoint requires SSD streaming where Flash Q2 fits resident.
Similar hardware alone does not imply equal throughput. Dividing Flash's speed
by the active-parameter ratio is only a rough sanity check; it ignores attention,
quantized kernels, memory traffic, and SSD misses.

Performance investigations use the existing native `ds4-bench` and its Metal
encoder-timeline diagnostic. Instrumented kernel timings are kept separate
from uninstrumented throughput. Cache size, cold/warm state, prompt, context,
sampling, source revision, and full-vocabulary comparisons must accompany
candidate measurements. Reported selected-expert synchronization time includes
GPU work before routing readback; it is not SSD read time. Likewise, the overall
cache-hit percentage includes reuse within prefill batches and is not a decode
cache-hit measurement.

The first profiling pass attributed 57.5% of decode GPU encoder time and 64.5%
of prefill GPU encoder time to general linear projections. A SIMD-group packing
experiment preserved 7,475,776 full-checkpoint logits exactly, but its balanced
comparison measured 5.53 versus 5.60 tokens/s. That production change was
discarded. The shared decode comparison now reserves both of its sessions in
memory admission before sizing the expert cache.

Muse's Q6_K projections now reuse packed bytes and scale metadata within each
block, preserving each lane's serial FP32 FMA order and the existing SIMD
reduction. The warmed synthetic 8192-by-8192 comparison measured 2.41x, 3.52x,
and 3.77x kernel speedups at 1, 16, and 512 input rows, with every output bit
identical. Reproduce this bounded check without loading a checkpoint:

```sh
make tests/test_muse_metal
./tests/test_muse_metal --bench-linear DS4_METAL_MUSE_Q6_REFERENCE 1.5
```

A balanced full-checkpoint decode comparison measured 7.02 tokens/s with the
new kernel versus 5.56 with the original, comparing all 7,475,776 logits and
36 selected IDs exactly. This used a 16-token prefix, four warmup steps, and 32
measured tokens per variant with two sessions sharing one engine/cache. It is
not the sustained 8K target. A standalone 512-token prompt measured 10.66 prefill
and 5.95 decode tokens/s; all 202,048 frontier logits matched the saved original
prefill exactly. Its admitted expert cache was 54.42 GiB versus 55.37 GiB in the
earlier 4.58/2.96 observation, so the controlled same-engine comparison is the
stronger attribution of speedup. Metal, quantization, graph, KV/session, memory,
and long-attention regressions passed.

The subsequent sustained sweep uses the published literary text, 2048-token
continued-prefill intervals, 128 greedy tokens per frontier, and default
512-token prefill chunks. A fixed 4096-expert cache (49.50 GiB) keeps its memory
allowance reproducible. On the Q6_K build `d6aec0e`:

| Context | Prefill tokens/s | Decode tokens/s |
| ---: | ---: | ---: |
| 2048 | 13.36 | 5.09 |
| 4096 | 6.65 | 4.31 |
| 6144 | 7.92 | 4.29 |
| 8192 | 7.56 | 4.32 |

The sweep passed in 1122.2 seconds; 57 memory samples peaked at 69.94 GiB RSS.
System Swapouts increased by 152 16-KiB pages (2.38 MiB). The final frontier
does not save a snapshot because there is no following prefill interval.
These results supersede the short-run speeds as the current sustained result;
10 decode tokens/s remains a target. They do not form a controlled speedup ratio
against the earlier archive-text, teacher-forced run with a larger expert cache.

An additional 8192-token profile attributes 22.5% of decode GPU encoder time to
Q6_K projections, 22.5% to Q8 projections, 28.3% to experts, 11.3% to attention,
and 9.4% to input normalization. Its 64-token decode took about 229 ms/token,
including about 168 ms/token in GPU encoders. Its 202,048 prompt-frontier logits
matched the uninstrumented sweep exactly. These instrumented proportions guide
investigation; use the uninstrumented sweep for throughput claims.

A partial process-I/O sampler covering
the 4K/6K/8K probes recorded approximately 22.3/31.8/27.6 GiB of attributed disk
reads across each roughly 29-second probe interval. Snapshot and sampling
boundaries make those approximate per-probe observations.

Gain-before-RMS normalization now loads four strided values ahead of its serial
FP32 accumulation. It preserves the multiplication/FMA order, SIMD reduction,
and in-place behavior. A warmed bounded comparison measured 2.87x at width 8192
with one row, 2.20x at width 4096 with eight rows, and 1.36x at width 8192 with
512 rows. Every output bit matched the original kernel. Tail-width fixtures
also compare both implementations against independent double-precision sums.

```sh
make tests/test_muse_metal
./tests/test_muse_metal --bench-norm 1.5
```

A balanced full-checkpoint comparison with a 16-token prefix, four warmup steps,
and 64 measured tokens per variant measured 7.87 versus 7.59 decode tokens/s.
All 13,941,312 logits and 68 selected IDs matched exactly. This short-prefix
comparison establishes a local gain; it does not replace the sustained 8K
measurement above. Q8 threadgroup sizing, a one-token Q8 specialization, and
grouping attention query heads by their shared KV head preserved exact outputs
but showed no useful model or kernel gain, so those production experiments
were discarded.


On the normalization build `89383b4`, the same fixed-cache sweep completed in
1242.6 seconds, with all 808,192 saved frontier logits exactly matching the
Q6_K baseline:

| Context | Prefill tokens/s | Decode tokens/s |
| ---: | ---: | ---: |
| 2048 | 8.00 | 3.39 |
| 4096 | 6.76 | 4.53 |
| 6144 | 7.14 | 4.28 |
| 8192 | 7.81 | 4.89 |

The [CSV](../speed-bench/m5_max_muse_ud_iq2_xxs.csv) records all 128-token probes.
The 63 memory samples peaked at 70.01 GiB RSS, with no increase in the system
Swapouts counter. This was the 14-inch M5 Max with 18 CPU/40 GPU cores, on AC
in Automatic power mode. Hardware state varied between sweeps; these numbers
are observations, not a controlled 13% attribution to normalization. The
balanced comparison above isolates its smaller gain. A host compilation ran
during the last prefill interval and finished before the final decode probe.
The latest standalone fixed-cache observation is 4.89 tokens/s; the 10 tokens/s target is
still unmet.

Decode now uses the existing Metal routing event to submit the independent
latent projection alongside routing. CPU expert reads start once routing is
ready, while the GPU can continue the latent work. This removes a submission
round trip without changing kernels, routing, or expert cache capacity.
A diagnostic rollback, `DS4_METAL_MUSE_LATENT_REFERENCE=1`, restores the earlier
schedule for balanced comparison. Bounded fixtures cover GPU-produced routing,
cache hits, forced reuse, and failure cleanup before and after read submission.
Independent full graphs and session/memory regressions pass.

With a 16-token prefix and 64 measured tokens per variant, the new schedule
measured 8.19 versus 7.87 tokens/s, with all 13,941,312 logits and 68 selected
IDs exact. At a 512-token prefix and 128 measured tokens, it measured 7.88
versus 7.27 tokens/s, with all 26,872,384 logits and 132 selected IDs exact.
Both runs used four warmup steps, two sessions sharing one engine/cache, and
`--clone-prefix` in the native decode A/B harness. These observations establish
the scheduling gain at short prefixes; they are not sustained 8K results.


The long-context schedule A/B on `2030616` used an 8192-token prefix,
8325-token allocation, four warmup steps, and 128 measured tokens per variant.
It passed all 26,872,384 raw logits and 132 selected IDs exactly, measuring
4.257 versus 4.198 tokens/s. Its auto-admitted shared expert cache was
59.04 GiB; its 61 memory samples peaked at 79.45 GiB RSS, with 36 additional
16-KiB system Swapouts pages. This shared-cache A/B is not a standalone
throughput comparison with the fixed-cache sweep. Reported GPU frequency
varied from 447 to 692 MHz across the timed loop; that observation does not
establish a thermal or power-limit cause. The connected adapter reports 100 W.

A restored 16-token prefix with four warmups measured 7.57 versus 7.73 tokens/s;
with 32 warmups and 256 measured tokens it measured 8.15 versus 7.66 tokens/s,
with all 58,391,872 logits and 288 selected IDs exact. Scheduling gains depend
on context and cache state; the standalone sustained target remains unmet.
Skipping an additional latent wait preserved correctness but gave only about
1% in its short comparison, so that extra path was removed.


Q6_K now exposes four blocks to the decode scheduler and reuses packed weights
across four input rows during prefill. Each row keeps the original lane mapping,
serial FP32 FMAs, and SIMD reduction. Independent SIMD groups pack output rows,
with the group count capped by the device pipeline limit. No dequantized weight
bank or additional tensor allocation is introduced. Partial input tiles and
partial output groups have strict output-bit and bounds-canary coverage.

```sh
make tests/test_muse_metal
./tests/test_muse_metal --bench-linear DS4_METAL_MUSE_Q6_LAYOUT_REFERENCE 1.05
```

Against the earlier optimized Q6_K kernel, the bounded comparison measured
1.09x at one input row, 1.50x/1.66x/1.40x at 3/4/5 rows, and 1.72x/1.74x/1.69x
at 8/16/512 rows. Every output bit matched. A 16-token restored-prefix decode
A/B, with 32 warmups and 256 measured tokens per variant, measured 7.88 versus
7.70 tokens/s with all 58,391,872 logits and 288 selected IDs exact. A balanced
ABBA prefill comparison of four fresh 512-token prompts measured 13.61 versus
12.77 tokens/s, with all 808,192 frontier logits exact. Its chunk size was 512,
context allocation 4096, and shared auto-admitted expert cache 59.80 GiB.
These are component and short-prefix observations; the sustained 8K target
still needs further optimization. Wider unrolls and eight-input tiles were
measured and rejected in favor of this layout.


IQ2_XXS experts now share packed weight decoding across up to four inputs.
The existing per-expert pair-list builder groups token/slot pairs; each input
keeps its original serial FP32 dot, SIMD reduction, and output slot. The active
expert addresses and buffer lifetime remain owned by the bounded SSD cache.
Output SIMD groups pack up to eight rows within the pipeline limit, while
independent input tiles avoid long serial loops for popular experts. The graph
reserves 4 MiB of pair lists plus 1 KiB of counts at a 512-token chunk, included
in admission and runtime accounting. Single-token expert execution is unchanged.

```sh
make tests/test_muse_quant
./tests/test_muse_quant --bench-experts
```

The bounded 4096-wide expert comparison measured 1.20x/1.49x/1.50x/1.75x/1.54x
at 2/3/5/16/512 inputs, with every intermediate output bit unchanged. One input
measured 1.00x. Fixtures cover duplicate selected IDs, rotated slot ranks, tails,
small-cache splitting, and exact full-graph results between streaming cache
budgets at the same chunk size. The existing resident F32 path uses a different
dot kernel and retains its independent numerical-reference check.

A full-checkpoint ABBA+BAAB comparison of eight fresh 512-token prompts measured
12.28 prefill tokens/s with grouped experts versus 9.88 with the prior kernel,
about 24% faster. All 1,616,384 frontier logits matched bit-for-bit. Context
allocation was 4096, chunk size 512, and the shared auto-admitted cache 60.21 GiB.
An earlier four-pass comparison measured 10.24 versus 11.85 and did not confirm
a gain; its grouped pass timings ranged from 28.74 to 71.27 seconds. Both runs
are retained in the validation artifacts. Hardware variation makes a single
pass unreliable. These are prefill results, not evidence of reaching the
10 tokens/s sustained decode target.


Further single-token IQ2 experiments were rejected: two- and four-block
unrolling initially measured about 5% faster but the gain did not repeat;
eight-block unrolling was about 30% slower. Replacing the sign lookup with its
exact parity formula was also slower. All retained raw-bit comparisons passed;
these experiments introduce no production kernel changes.

The existing decode A/B benchmark can emit its actual prefix and evaluated
input token IDs with `DS4_METAL_DECODE_BENCH_TOKEN_TRACE=1`. It also records
individual evaluation durations after each timed interval. Recorded durations
are checked against printed aggregates and support paired timing analysis.
This diagnostic is outside timed evaluation and selection. It allows prefix-copy hit rates to be
measured before adding stateful lookahead. It does not change sampling or model
execution. Saved prefixes remain usable through `--load-prefix` for subsequent
native decode experiments.


The updated kernels also completed an 8192-token prefill followed by 32 warmup
and 256 measured decode tokens per variant. With context allocation 8481 and a
shared auto-admitted 58.09 GiB expert cache, current Q6 measured 4.33 versus
4.30 tokens/s for its prior layout. All 58,391,872 logits and 288 selected IDs
matched bit-for-bit. Runtime was 821.68 seconds, peak RSS 78.88 GiB, and the
system swapout counter did not increase. The saved native prefix payload is
1,020,056,884 bytes; all 202,048 saved frontier logits exactly match the earlier
8192-token reference. A subsequent disk restore preserved the complete 288-token
trajectory. These paired-session observations use a different cache and warmup
protocol from the standalone CSV above.

Loading that saved state and repeating 256 measured tokens without profiling
measured 8.40 and 8.46 tokens/s for two identical decode paths. A longer run of
1024 measured tokens per path averaged 5.51 tokens/s, with all 213,564,736 logits
and 1,056 selected IDs exact. It loaded the same 8192-token state into a
9249-token allocation; its first 288 evaluated IDs exactly matched the shorter
run. Peak RSS was 79.99 GiB and the system swapout counter again did not
increase. Approximate successive window rates declined from 8 to 7 to 5 tokens/s.
The short 8.4 result therefore does not establish sustained 10 tokens/s.

Telemetry shows different GPU conditions: active samples after long prefill
averaged about 473 MHz and 6.88 W, versus 1318 MHz and 31.96 W across the shorter
restored run. Decode boundaries for the former are approximated by the saved
prefix file's completion time. These observations are correlated with speed;
they do not establish why clocks changed. The Mac remained in Automatic power
mode, with no power, fan, or sleep settings changed.

A fresh restored-state profile attributes about 27.1% of GPU time to Q6,
26.0% to Q8, 17.0% to expert gate/up, 9.0% to expert down, 10.0% to split
attention, and 4.1% to input normalization. Public Metal counter enumeration on
this device exposes timestamps only. The instrumented run is diagnostic, not a
throughput target measurement.

Causal prefix-copy analysis of the 288 actual evaluated inputs found no
8-token suffix opportunities. A looser 3-token match saved at most four model
forwards while adding 38 extra input rows. This trace does not justify a
stateful lookahead path. A bounded four-output-row Q8 weight-layout experiment
preserved every output bit, but gained only about 4% on the largest decode
matrix while regressing smaller shapes by about 9–11%; no production weight
repacking is introduced. Disabling existing advisory expert readahead measured
8.03 versus 8.05 tokens/s, with all 58,391,872 logits exact, so its default is
unchanged. A diagnostic user-initiated process activity did not prevent the
long-run slowdown (about 5.2 tokens/s); no activity policy is added to inference.


Single-token IQ2 gate/up evaluation now shares input loads between two
independent original FP32 chains. It keeps the same per-lane values, serial
FMAs, SIMD sums, and sigmoid gate; it adds no weight bank or scratch tensor.
The existing backend MoE pair kernels use a different accumulation layout,
so this specialization preserves Muse's row-dot ordering. A diagnostic
`DS4_METAL_MUSE_IQ2_PAIR_REFERENCE=1` selects the prior kernel.

The bounded combined-expert measurement improved by about 9.6%, just below
its exploratory 10% target. A 256-token native comparison measured 8.00 versus
7.83 tokens/s, but its timing uncertainty included zero. Extending the paired
comparison to 1024 measured tokens per variant established 5.44 versus 5.25
tokens/s, about 3.59% faster, with a within-run 95% block-bootstrap interval of
2.96–4.20%. All 213,564,736 logits and 1,056 selected IDs matched bit-for-bit.
The uncertainty estimate resamples 16-step blocks with a fixed seed and 10,000
replicates; it does not promise the same gain under different hardware
conditions. Focused quantization, experts, complete graph, sessions, memory,
and 8193-token attention checks pass, and all native binaries build.


A bounded IQ2 lookup-table staging experiment preserved every intermediate
bit but improved only about 2–3% over the comparable constant-table kernels;
it is not retained. A separate isolation test uses the same native kernels,
cache buffers, weights, and argument layout across 12 rotated rounds of 16
repetitions. Full expert preparation measured 0.4684 ms/call, prebound resources
with the same per-call submissions measured 0.4388 ms, and prebound dispatches
in one batch measured 0.2944 ms. This bounds preparation overhead at about
30 microseconds and attributes about 144 microseconds to the different
submission schedule in that cache-hot synthetic workload. It is not a native
model speedup estimate, and does not establish the cause of lower long-run GPU
clocks. No new cache planner or GPU rendezvous mechanism is introduced.


On `ad97c58`, the same standalone fixed-cache protocol completed in 821.76
seconds. All four prompt frontiers, totaling 808,192 float32 logits, matched
the normalization baseline bit-for-bit. Forty-two memory samples peaked at
69.52 GiB RSS; the system swapout counter did not increase. The current CSV
records this completed sweep:

| Context | Prefill tokens/s | Decode tokens/s |
| ---: | ---: | ---: |
| 2048 | 14.75 | 5.11 |
| 4096 | 11.46 | 4.50 |
| 6144 | 10.86 | 4.48 |
| 8192 | 10.67 | 4.38 |

The earlier normalization sweep remains documented above, including its
4.89 tokens/s result at 8K. These separate sweeps show stronger prefill but
no consistent decode gain across hardware conditions. They do not invalidate
the within-run exact IQ2 pair comparison, and that paired improvement does
not imply the standalone 10 tokens/s goal has been reached.


Two native 256-token tests of the existing TP keep-alive worker were rejected:
one threadgroup measured 6.47 versus 7.06 tokens/s without it, and four groups
measured 6.56 versus 7.33. Each test compared all 58,391,872 logits and 288
selected IDs exactly; peak RSS stayed below 79 GiB and swapouts did not
increase. Setup and worker teardown were outside timed evaluation. A bounded
pre-encoded shared-event handshake test also measured about the same as
ordinary per-call submissions (0.459 versus 0.454 ms); flag arrival with event
release was slower. These component measurements do not support adding a
new GPU rendezvous protocol. No keep-alive inference policy is introduced.

Increasing active expert-read workers from nine to eighteen measured 8.66
versus 8.71 tokens/s over 256 native steps, with all logits and selected IDs
exact; its paired uncertainty included no difference. Moving the same Q8
bytes into GPU-private storage preserved every output bit but did not improve
the largest projection. Neither experiment changes the default runtime.
Extending causal suffix-copy analysis to all 1,056 evaluated inputs found at
most nine saved forwards while adding 195 input rows, so that longer trace
also does not justify speculative prefix-copy execution.


Muse now evaluates its independent Q/K/V/gate projections in one bounded
concurrent compute pass on M5 for one to four input rows. It reuses the
original Q6, Q8 and F32 kernels and the existing command batch. Output/input
alias checks allow disjoint views of one buffer, reject overlaps, and preserve
the caller's batch after failure. The pass always closes before a dependent
consumer. Larger batches and other devices retain serial dispatch. Set
`DS4_METAL_MUSE_PROJECTIONS_REFERENCE=1` for the prior serial schedule.

A bounded 137.5-MiB test of actual checkpoint projections measured an 8–9%
component improvement for one to four rows with every output bit exact.
The native saved-8K comparison then completed 32 warmup and 1024 measured
steps per path: concurrent dispatch measured 4.8897 versus 4.7940 tokens/s,
a 2.00% improvement with a within-run 95% block-bootstrap interval of
1.32–2.65%. All 213,564,736 logits and 1,056 selected IDs matched bit-for-bit.
The 440.93-second run had 23 memory samples, peak RSS 79.19 GiB and no
increase in system swapouts. This retains a small demonstrated gain, while
the sustained 10 tokens/s target remains unmet. The standalone CSV above
continues to identify the earlier `ad97c58` sweep rather than mixing protocols.

The new `test-muse-projections` target covers mixed quantization, irregular
output widths, one to four projections, shared-buffer views, a dependent
consumer, partial-failure recovery, and serial rollback. Quantization, complete
graph, session, memory and 8193-token attention checks pass, and every native
binary builds. Pipeline fast lookup and fixed/slab residency-set hints were
also tested independently; their sub-1% native differences were inconclusive,
so those policies are not changed.
