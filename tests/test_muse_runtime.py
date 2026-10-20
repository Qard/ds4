"""Metal parity tests against MLX's unmodified Muse graph, using tiny weights."""

import json
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map
from mlx_lm.models.muse_spark import Model, ModelArgs

from muse.runtime import load_streamed_model


def tiny_checkpoint(root):
    config = dict(
        model_type="muse_spark",
        text_config=dict(
            hidden_size=128, num_attention_heads=2, num_key_value_heads=1,
            head_dim=64, num_hidden_layers=2, vocab_size=256,
            rms_norm_eps=1e-5, post_norm_eps=1e-8,
            rope_parameters={"rope_theta": 500000.0},
            layer_rope_theta=[500000.0, 0],
            layer_types=["sliding_attention", "full_attention"],
            sliding_window=4, final_logit_softcapping=20.0,
            output_multiplier=0.17677669529663687,
            num_local_experts=4, num_experts_per_tok=2,
            moe_hidden_size=128, moe_intermediate_size=128,
        ),
        quantization={"bits": 3, "group_size": 32, "mode": "affine"},
    )
    mx.random.seed(7)
    model = Model(ModelArgs.from_dict(config))
    model.update(tree_map(lambda x: x.astype(mx.bfloat16), model.parameters()))
    for layer in model.layers:
        layer.mlp.gate.weight = layer.mlp.gate.weight.astype(mx.float32)
        layer.mlp.gate.e_score_correction_bias = mx.array([0.0, 0.2, 0.1, 0.0])

    def predicate(path, module):
        if not hasattr(module, "to_quantized") or path == "lm_head":
            return False
        params = dict(group_size=32, bits=3 if ".experts." in path else 4, mode="affine")
        # Mixed widths ensure the offloader uses each projection's own config.
        if path.endswith("experts.down_proj"):
            params["bits"] = 5
        config["quantization"][path] = params
        return params

    nn.quantize(model, group_size=32, bits=3, class_predicate=predicate)
    model.eval()
    mx.eval(model.parameters())
    weights = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(root / "model.safetensors"), weights)
    (root / "config.json").write_text(json.dumps(config))
    return model, config


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.reference, self.config = tiny_checkpoint(self.root)

    def load(self, **kwargs):
        runner = load_streamed_model(self.root, cache_bytes=86016,
                                     context=32, scratch_bytes=1024**2,
                                     memory_bytes=128 * 1024**2, **kwargs)
        self.addCleanup(runner.close)
        return runner

    def assertLogitsClose(self, expected, actual):
        mx.eval(expected, actual)
        error = mx.max(mx.abs(expected - actual)).item()
        self.assertLess(error, 0.015, f"logits differ by {error}")

    def test_prefill_and_decode_match_reference_through_cache_evictions(self):
        runner = self.load()
        reference_cache, streamed_cache = self.reference.make_cache(), runner.model.make_cache()
        for ids in ([1, 2, 3], [4, 5], [6], [7], [8]):
            tokens = mx.array([ids])
            expected = self.reference(tokens, cache=reference_cache)
            actual = runner.model(tokens, cache=streamed_cache)
            self.assertLogitsClose(expected, actual)
        self.assertGreater(runner.experts.stats["evictions"], 0)
        self.assertGreater(runner.experts.stats["hits"], 0)
        self.assertLessEqual(runner.experts.bytes_used, 86016)
        self.assertFalse(any("experts.gate_proj" in name for name, _ in
                             tree_flatten(runner.model.parameters())))

    def test_single_expert_budget_can_process_wide_prefill(self):
        runner = load_streamed_model(self.root, cache_bytes=28672,
                                     context=32, scratch_bytes=1024**2,
                                     memory_bytes=128 * 1024**2)
        self.addCleanup(runner.close)
        tokens = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
        self.assertLogitsClose(self.reference(tokens), runner.model(tokens))

    def test_missing_resident_weight_fails_instead_of_running_random_weights(self):
        weights = mx.load(str(self.root / "model.safetensors"))
        del weights["model.language_model.layers.0.input_layernorm.weight"]
        mx.save_safetensors(str(self.root / "model.safetensors"), weights)
        with self.assertRaisesRegex(ValueError, "Missing.*input_layernorm"):
            self.load()

    def test_memory_plan_refuses_overcommit(self):
        with self.assertRaisesRegex(ValueError, "memory|budget"):
            load_streamed_model(self.root, cache_bytes=86016, context=32768,
                                memory_bytes=1024, scratch_bytes=1024)

    def test_rejects_raw_nvfp4_before_loading_payloads(self):
        self.config.pop("quantization")
        self.config["quantization_config"] = {"quant_method": "modelopt"}
        (self.root / "config.json").write_text(json.dumps(self.config))
        with self.assertRaisesRegex(ValueError, "converted|affine"):
            self.load()

    def test_rejects_integer_norm_instead_of_accepting_invalid_kernel_inputs(self):
        weights = mx.load(str(self.root / "model.safetensors"))
        name = "model.language_model.layers.0.input_layernorm.weight"
        weights[name] = weights[name].astype(mx.uint32)
        mx.save_safetensors(str(self.root / "model.safetensors"), weights)
        with self.assertRaisesRegex(ValueError, "dtype"):
            self.load()

    def test_rejects_unconfigured_expert_layer(self):
        weights = mx.load(str(self.root / "model.safetensors"))
        name = "model.language_model.layers.0.mlp.experts.gate_proj.weight"
        weights[name.replace("layers.0", "layers.2")] = weights[name]
        mx.save_safetensors(str(self.root / "model.safetensors"), weights)
        with self.assertRaisesRegex(ValueError, "layer"):
            self.load()


if __name__ == "__main__":
    unittest.main()
