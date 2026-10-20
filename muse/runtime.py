"""Use MLX's Muse graph while keeping routed expert banks exclusively on SSD.

The experimental path accepts converted MLX affine checkpoints only. It never
calls the ordinary whole-model loader or evaluates a full expert bank. The
conservative first implementation synchronizes at expert-group boundaries;
this bounds memory and gives a correctness baseline for later Metal overlap.
"""

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm.models.activations import precise_swiglu
from mlx_lm.models.muse_spark import Model, ModelArgs

from .storage import EXPERT_RE, ExpertCache, TensorStore


GIB = 1024**3
PREFIX = "model.language_model.layers."
SKIP_PREFIXES = ("model.vision_tower.", "model.vision_adapter.",
                 "model.vision_projection.", "model.audio_tower.", "perception_emb_norm.")
NUMPY_DTYPES = {"F16": "<f2", "BF16": "<u2", "F32": "<f4", "F64": "<f8",
                "U8": "u1", "I8": "i1", "U16": "<u2", "I16": "<i2",
                "U32": "<u4", "I32": "<i4", "U64": "<u8", "I64": "<i8",
                "BOOL": "?"}


def read_array(store, name, row=None):
    info = store.tensors[name]
    shape = info.shape if row is None else info.shape[1:]
    # mx.array copies the buffer: no view can keep an entire shard mapped.
    data = np.frombuffer(store.read(name, row=row), dtype=NUMPY_DTYPES[info.dtype])
    array = mx.array(data.reshape(shape))
    if info.dtype == "BF16":
        array = array.view(mx.bfloat16)
    mx.eval(array)
    return array


def quant_params(config, path):
    quant = config.get("quantization")
    if not isinstance(quant, dict) or "bits" not in quant or "group_size" not in quant:
        raise ValueError("Use a converted MLX affine checkpoint, not raw BF16/NVFP4 or GGUF")
    params = {k: quant[k] for k in ("bits", "group_size", "mode") if k in quant}
    params.update(quant.get(path, {}))
    params.setdefault("mode", "affine")
    if (params["mode"] != "affine" or params["bits"] not in (2, 3, 4, 5, 6, 8)
            or params["group_size"] not in (32, 64, 128)):
        raise ValueError(f"Unsupported affine quantization at {path}: {params}")
    return params


@dataclass(frozen=True)
class MemoryPlan:
    memory_bytes: int
    resident_bytes: int
    kv_bytes: int
    scratch_bytes: int
    expert_cache_bytes: int
    largest_expert_bytes: int
    checkpoint_bytes: int

    def as_dict(self):
        return asdict(self)


def memory_plan(store, config, context, cache_bytes=None, memory_bytes=None,
                scratch_bytes=4 * GIB):
    if type(context) is not int or context <= 0 or scratch_bytes < 0:
        raise ValueError("Context must be positive and scratch memory nonnegative")
    if config.get("model_type") != "muse_spark":
        raise ValueError("Expected model_type=muse_spark")
    quant_params(config, "")
    text = config["text_config"]
    layers = text["num_hidden_layers"]
    if (len(text["layer_types"]) != layers or len(text["layer_rope_theta"]) != layers
            or text["num_local_experts"] <= 0
            or not 0 < text["num_experts_per_tok"] <= text["num_local_experts"]
            or any(t not in ("sliding_attention", "full_attention") for t in text["layer_types"])):
        raise ValueError("Inconsistent Muse layer/expert configuration")
    if context > text.get("max_position_embeddings", 1048576):
        raise ValueError("Context exceeds model position limit")
    resident = sum(info.nbytes for name, info in store.tensors.items()
                   if not EXPERT_RE.fullmatch(name) and not name.startswith(SKIP_PREFIXES))
    full = text["layer_types"].count("full_attention")
    # KVCache grows in 256-token steps; use the actual rounded capacity.
    full_capacity = math.ceil(context / 256) * 256
    window_capacity = math.ceil(min(context, text["sliding_window"]) / 256) * 256
    kv = (2 * text["num_key_value_heads"] * text["head_dim"] * 2
          * (full * full_capacity + (layers - full) * window_capacity))
    largest = max((store.expert_bytes(layer) for layer in range(layers)), default=0)
    if not largest:
        raise ValueError("No converted routed experts found")
    # Leave 20% of the device's recommended set for OS/other processes, then
    # account for mandatory model/KV/scratch before admitting an expert cache.
    if memory_bytes is None:
        memory_bytes = int(mx.device_info()["max_recommended_working_set_size"] * 0.8)
    available = memory_bytes - resident - kv - scratch_bytes
    cache_bytes = available if cache_bytes is None else cache_bytes
    if cache_bytes < largest or cache_bytes > available:
        raise ValueError(
            f"Expert cache exceeds memory budget or cannot fit one expert: "
            f"requested={cache_bytes}, available={available}, largest={largest}")
    return MemoryPlan(memory_bytes, resident, kv, scratch_bytes, cache_bytes,
                      largest, sum(info.nbytes for info in store.tensors.values()))


def _at(model, path):
    node = model
    for part in path.split("."):
        node = node[int(part)] if isinstance(node, list) else node[part]
    return node


def _set(model, path, value):
    parent, _, key = path.rpartition(".")
    node = _at(model, parent) if parent else model
    if isinstance(node, list):
        node[int(key)] = value
    else:
        node[key] = value


class StreamedExperts(nn.Module):
    def __init__(self, layer, norm, cache, config, store):
        super().__init__()
        self.post_expert_norm = norm
        self._layer, self._cache = layer, cache
        self._bytes = store.expert_bytes(layer)
        self._params = {p: quant_params(config, f"{PREFIX}{layer}.mlp.experts.{p}")
                        for p in ("gate_proj", "up_proj", "down_proj")}

    def _linear(self, x, expert, projection):
        tensors = expert[projection]
        return mx.quantized_matmul(x, tensors["weight"], tensors["scales"],
                                   tensors["biases"], transpose=True,
                                   **self._params[projection])

    def __call__(self, x, indices):
        # Router results must be known before disk reads. Evaluate x too, so
        # earlier layers release dependencies before this cache can evict them.
        mx.eval(x, indices)
        ids = np.array(indices).reshape(-1, indices.shape[-1])
        flat = x.reshape(-1, x.shape[-1])
        output = mx.zeros((flat.shape[0], ids.shape[1], flat.shape[-1]), dtype=x.dtype)
        selected = np.unique(ids).tolist()
        per_group = max(1, self._cache.budget // self._bytes)
        for start in range(0, len(selected), per_group):
            group = selected[start:start + per_group]
            keys = [(self._layer, eid) for eid in group]
            with self._cache.lease(keys) as experts:
                for eid in group:
                    rows, slots = np.where(ids == eid)
                    row_ids, slot_ids = mx.array(rows.astype(np.int32)), mx.array(slots.astype(np.int32))
                    inputs = flat[row_ids]
                    weights = experts[(self._layer, eid)]
                    up = self._linear(inputs, weights, "up_proj")
                    gate = self._linear(inputs, weights, "gate_proj")
                    hidden = precise_swiglu(up, gate, up)
                    down = self._linear(hidden, weights, "down_proj")
                    output = output.at[row_ids, slot_ids].add(down)
                # GPU commands must finish using leased weights before eviction.
                mx.eval(output)
                del weights, up, gate, hidden, down, inputs
            del experts
        return output.reshape(*indices.shape, x.shape[-1])


@dataclass
class StreamedModel:
    model: Model
    experts: ExpertCache
    store: TensorStore
    plan: MemoryPlan

    def close(self):
        self.experts.close()
        self.store.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def _validate_experts(store, config):
    text = config["text_config"]
    count = text["num_local_experts"]
    if any(int(EXPERT_RE.fullmatch(name)[1]) >= text["num_hidden_layers"]
           for name in store.tensors if EXPERT_RE.fullmatch(name)):
        raise ValueError("Expert tensors reference an unconfigured layer")
    for layer in range(text["num_hidden_layers"]):
        names = set(store.expert_tensors(layer))
        expected = set()
        for projection in ("gate_proj", "up_proj", "down_proj"):
            path = f"{PREFIX}{layer}.mlp.experts.{projection}"
            params = quant_params(config, path)
            inp, out = text["moe_hidden_size"], text["moe_intermediate_size"]
            if projection == "down_proj":
                inp, out = out, inp
            for suffix, shape in (
                ("weight", (count, out, inp * params["bits"] // 32)),
                ("scales", (count, out, inp // params["group_size"])),
                ("biases", (count, out, inp // params["group_size"])),
            ):
                name = f"{path}.{suffix}"
                expected.add(name)
                if name not in store.tensors:
                    raise ValueError(f"Missing converted expert tensor: {name}")
                info = store.tensors[name]
                if info.shape != shape or (suffix == "weight" and info.dtype != "U32"):
                    raise ValueError(f"Invalid expert shape/dtype: {name}: {info}")
                if suffix != "weight" and info.dtype not in ("BF16", "F16", "F32"):
                    raise ValueError(f"Invalid expert scale dtype: {name}")
        if names != expected:
            raise ValueError(f"Unexpected expert layout in layer {layer}")


def load_streamed_model(root, *, cache_bytes=None, context=32768, workers=4,
                        memory_bytes=None, scratch_bytes=4 * GIB):
    root = Path(root)
    config = json.loads((root / "config.json").read_text())
    # Validate the format before creating any model arrays.
    quant_params(config, "")
    store = TensorStore(root)
    cache = None
    try:
        plan = memory_plan(store, config, context, cache_bytes, memory_bytes, scratch_bytes)
        _validate_experts(store, config)

        def load_expert(key):
            layer, eid = key
            result = {}
            for name in store.expert_tensors(layer):
                projection, suffix = name.split(".")[-2:]
                result.setdefault(projection, {})[suffix] = read_array(store, name, row=eid)
            return result

        cache = ExpertCache(plan.expert_cache_bytes, lambda k: store.expert_bytes(k[0]),
                            load_expert, workers=workers)
        # Constructors are lazy. Remove all random expert banks before evaluating
        # anything; their hundreds of billions of parameters never materialize.
        model_config = {**config, "vision_config": None}
        model = Model(ModelArgs.from_dict(model_config))
        for layer, block in enumerate(model.layers):
            block.mlp.experts = StreamedExperts(layer, block.mlp.experts.post_expert_norm,
                                               cache, config, store)

        # Record logical shapes, not arrays, so replacement frees initializer DAGs.
        logical = {name: value.shape for name, value in tree_flatten(model.parameters())}
        for path, module in list(model.named_modules()):
            if not isinstance(module, (nn.Linear, nn.Embedding)):
                continue
            if path + ".scales" not in store.tensors:
                continue
            params = quant_params(config, path)
            shape = logical[path + ".weight"]
            # A tiny constructor supplies MLX's public quantized module behavior
            # without ever quantizing random model-sized initializers.
            if isinstance(module, nn.Embedding):
                replacement = nn.QuantizedEmbedding(1, params["group_size"], **params)
                replacement.num_embeddings, replacement.dims = shape
            else:
                replacement = nn.QuantizedLinear(params["group_size"], 1,
                                                  bias="bias" in module, **params)
            _set(model, path, replacement)
            logical[path + ".weight"] = (shape[0], shape[1] * params["bits"] // 32)
            logical[path + ".scales"] = (shape[0], shape[1] // params["group_size"])
            logical[path + ".biases"] = logical[path + ".scales"]

        expected = set(logical)
        actual = {name for name in store.tensors
                  if not EXPERT_RE.fullmatch(name) and not name.startswith(SKIP_PREFIXES)}
        if expected - actual:
            raise ValueError(f"Missing resident tensors: {sorted(expected - actual)}")
        if actual - expected:
            raise ValueError(f"Unexpected tensors; use a converted MLX checkpoint: {sorted(actual - expected)}")
        for name in sorted(expected):
            info = store.tensors[name]
            if info.shape != logical[name]:
                raise ValueError(f"Invalid resident shape: {name}: expected {logical[name]}")
            packed = name.endswith(".weight") and name[:-7] + ".scales" in expected
            if (packed and info.dtype != "U32") or (
                    not packed and info.dtype not in ("BF16", "F16", "F32")):
                raise ValueError(f"Invalid resident dtype: {name}: {info.dtype}")
        # Validate the entire resident directory before reading even one payload.
        for name in sorted(expected):
            _set(model, name, read_array(store, name))
        model.eval()
        mx.eval(model.parameters())
        return StreamedModel(model, cache, store, plan)
    except BaseException:
        if cache is not None:
            cache.close()
        store.close()
        raise
