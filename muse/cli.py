"""Text-only experimental runner; local, converted MLX checkpoints only."""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm.sample_utils import make_sampler
from mlx_lm.utils import load_tokenizer

from .runtime import GIB, load_streamed_model, memory_plan
from .storage import TensorStore


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def gib(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a positive, finite GiB value")
    return int(number * GIB)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-m", "--model", required=True, type=Path, help="Local MLX checkpoint directory")
    prompts = p.add_mutually_exclusive_group()
    prompts.add_argument("-p", "--prompt")
    prompts.add_argument("--prompt-file", type=Path)
    prompts.add_argument("--token-ids", help="Comma-separated IDs; bypass tokenizer for numeric validation")
    p.add_argument("--inspect", action="store_true", help="Validate headers and print memory plan without loading tensors")
    p.add_argument("--ctx", type=positive_int, default=32768)
    p.add_argument("--tokens", type=positive_int, default=128)
    p.add_argument("--prefill-chunk", type=positive_int, default=16)
    p.add_argument("--cache-gib", type=gib, help="Expert cache only; automatic after model/KV/scratch accounting")
    p.add_argument("--memory-gib", type=gib, help="Total planned working set; default 80%% of Metal recommendation")
    p.add_argument("--read-workers", type=positive_int, default=4)
    p.add_argument("--temp", type=float, default=0.0, help="Sampling temperature; default greedy for validation")
    p.add_argument("--reasoning-effort", choices=("low", "medium", "high", "xhigh"), default="low")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if not math.isfinite(args.temp) or args.temp < 0:
            raise ValueError("Temperature must be finite and nonnegative")
        config = json.loads((args.model / "config.json").read_text())
        with TensorStore(args.model) as store:
            plan = memory_plan(store, config, args.ctx, args.cache_gib, args.memory_gib)
        if args.inspect:
            print(json.dumps(plan.as_dict(), indent=2))
            return 0

        tokenizer = None
        if args.token_ids is not None:
            prompt = [int(token.strip()) for token in args.token_ids.split(",")]
        else:
            text = args.prompt_file.read_text() if args.prompt_file else args.prompt
            if not text:
                raise ValueError("Pass --prompt, --prompt-file, --token-ids, or --inspect")
            # Let the checkpoint's Jinja template take precedence over defaults.
            tokenizer = load_tokenizer(args.model)
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": text}], tokenize=True,
                add_generation_prompt=True, reasoning_effort=args.reasoning_effort)
        if not prompt or any(token < 0 or token >= config["text_config"]["vocab_size"] for token in prompt):
            raise ValueError("Prompt token IDs must be within the model vocabulary")
        if len(prompt) + args.tokens > args.ctx:
            raise ValueError(f"Prompt plus generation exceeds context: {len(prompt)} + {args.tokens} > {args.ctx}")

        print(json.dumps({"memory_plan": plan.as_dict()}), file=sys.stderr, flush=True)
        old_limit = mx.set_memory_limit(plan.memory_bytes)
        old_cache = mx.set_cache_limit(256 * 1024**2)
        try:
            mx.reset_peak_memory()
            started = time.monotonic()
            with load_streamed_model(args.model, cache_bytes=plan.expert_cache_bytes,
                                     context=args.ctx, workers=args.read_workers,
                                     memory_bytes=plan.memory_bytes) as runner:
                loaded = time.monotonic()
                cache = runner.model.make_cache()
                logits = None
                for start in range(0, len(prompt), args.prefill_chunk):
                    logits = runner.model(mx.array([prompt[start:start + args.prefill_chunk]]), cache=cache)
                    mx.eval(logits)
                    print(f"prefill {min(start + args.prefill_chunk, len(prompt))}/{len(prompt)}",
                          file=sys.stderr, flush=True)
                prefilled = time.monotonic()
                generated = []
                sampler = make_sampler(temp=args.temp)
                if tokenizer:
                    tokenizer.detokenizer.reset()
                for step in range(args.tokens):
                    logprobs = logits[:, -1, :] - mx.logsumexp(logits[:, -1, :], keepdims=True)
                    token = sampler(logprobs).item()
                    generated.append(token)
                    if tokenizer:
                        if token in tokenizer.eos_token_ids:
                            break
                        tokenizer.detokenizer.add_token(token)
                        print(tokenizer.detokenizer.last_segment, end="", flush=True)
                    if step + 1 < args.tokens:
                        logits = runner.model(mx.array([[token]]), cache=cache)
                        mx.eval(logits)
                if tokenizer:
                    tokenizer.detokenizer.finalize()
                    print(tokenizer.detokenizer.last_segment, flush=True)
                else:
                    print(json.dumps({"generated_token_ids": generated}))
                print(json.dumps({"load_seconds": loaded - started,
                                  "prefill_seconds": prefilled - loaded,
                                  "decode_seconds": time.monotonic() - prefilled,
                                  "peak_mlx_bytes": mx.get_peak_memory(),
                                  "expert_cache": runner.experts.stats}), file=sys.stderr)
        finally:
            mx.set_cache_limit(old_cache)
            mx.set_memory_limit(old_limit)
        return 0
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"ds4-muse: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
