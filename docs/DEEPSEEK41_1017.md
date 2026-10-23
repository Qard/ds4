# DeepSeek V4.1 Flash 1017 on a 128 GB Mac

The 1017 text checkpoint uses ds4's existing native V4.1 graph, tokenizer,
Engram reader and SSD expert cache. No Muse code or separate runner is required.
Use the Q2 file on a fast local SSD; leave the expert cache budget automatic.
The file is 340.60 GiB, including 188.83 GiB of disk-only Engram tables.
Its 151.77 GiB of main weights require SSD streaming on a single 128 GB Mac.

```sh
make -j4
DS4_GGUF_DIR="$HOME/.ds4/models/deepseek-v4.1-flash-1017" \
  ./download_model.sh ds41f-q2
./ds4 -m "$HOME/.ds4/models/deepseek-v4.1-flash-1017/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf" \
  --ssd-streaming --ctx 8192
```

The downloader resumes interrupted transfers through the official `hf` CLI,
then checks the complete file's size and SHA-256. It updates `ds4flash.gguf`
only after verification. Install `huggingface_hub` if `hf` is unavailable.
The same model path and streaming options work with `ds4-agent` and `ds4-server`.

## Artifact identity

- Publisher: [antirez/deepseek-v4.1-flash-gguf](https://huggingface.co/antirez/deepseek-v4.1-flash-gguf).
- Inspected publisher revision: `9dcee51692e6540f697141fc98b097e3c96492a3`.
- Source: [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017), revision `1bae0e1088993b9b82de08f59de5be5a34e42170`.
- File: `DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf`.
- Bytes: `365713686528`.
- SHA-256: `5296e85ff64b42ecb62c8d0c0c54e1caa07e3df8e93112f2365ebbaed62ce93a`.

The Q4 and vision downloader targets still select the original V4.1 checkpoint.
Do not pair its vision encoder or DSpark weights with 1017. This setup covers
text inference without speculative decoding. Correct 1017 vision matching and
a verified matching DSpark artifact are separate work.

## Validation before the full download

On the M5 Max, the published 1017 GGUF header passes the complete native model
layout check. A sparse zero-weight fixture with that actual header passes the
SSD graph/reset test, memory-budget admission tests, and public session/prefix/
snapshot/cancellation tests. Against the pinned official 1017 tokenizer and
encoder, 206 CLI prompts and 60 OpenAI/Anthropic prompts match exactly.

The native Metal component suite, CLI checks, prefill-dispatch checks, DSpark
candidate-mask checks, five conversion fixtures, two vision-conversion fixtures,
ten downloader tests, and server protocol suite pass. Candidate-mask tests do
not establish compatibility of any particular speculative checkpoint.

The initial default Metal suite exposed an existing FP4/E4M3 midpoint rounding
error: `9.375 / 3.75` could land above 2.5 with fast reciprocal division and
round to 11.25 instead of 7.5. The V4.1 quantizer now compares exact scaled
midpoints, retaining ties-to-even without a global safe-math switch. Explicit
positive/negative midpoint cases at seven E4M3 scales and the original random
round-trip oracle pass with default Metal math.

These checks establish format/protocol/component compatibility. They do not
establish real-weight output quality or throughput; those require the completed
verified checkpoint. No 1017 generation-speed result is claimed yet.

## First throughput run

Keep the lid open, use AC power, and record power mode and continuous telemetry.
Run a short generation before a longer sweep. For a comparable context sweep:

```sh
./ds4-bench -m "$HOME/.ds4/models/deepseek-v4.1-flash-1017/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf" \
  --ssd-streaming --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 --ctx-max 8192 --step-mul 2 --gen-tokens 128
```

Record effective cache size, memory use and cold versus warm behavior. The
published V4 Flash baseline uses different weights and architecture; it is not
a measured V4.1 Flash 1017 performance guarantee.
