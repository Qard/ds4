# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2,<3"]
# ///
"""Sparse production-shape fixtures: header-only allocation, no inference."""
import struct
import sys
import tempfile
import subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'gguf-tools'))
from qwen4_pack_to_qwen4exp import kv_bytes, w_str
from gguf_join import tensor_bytes


def fixture(path, bad=None):
    ints = {'block_count': 62, 'context_length': 1048576, 'embedding_length': 8192,
            'attention.head_count': 128, 'attention.head_count_kv': 16,
            'attention.key_length': 64, 'attention.value_length': 64,
            'expert_count': 256, 'expert_used_count': 8,
            'attention.sliding_window': 2048, 'moe_latent_size': 4096,
            'expert_feed_forward_length': 4096, 'expert_gating_func': 2}
    floats = {'rope.freq_base': 500000., 'rope.freq_base_swa': 500000.,
              'attention.layer_norm_rms_epsilon': 1e-5, 'post_norm_epsilon': 1e-8,
              'logit_scale': .1767766953, 'final_logit_softcapping': 20.,
              'expert_weights_scale': 1.}
    if bad == 'latent': ints['moe_latent_size'] = 8192
    kv = [('general.architecture', 8, 'muse-spark')]
    kv += [('muse-spark.' + k, 4, v) for k, v in ints.items()]
    kv += [('muse-spark.' + k, 6, v) for k, v in floats.items()]
    kv += [('muse-spark.expert_weights_norm', 7, True)]
    pattern = [int(i % 4 != 1) for i in range(62)]
    if bad == 'pattern': pattern.pop()
    # Arr is the common GGUF writer's typed-array representation.
    from qwen4_pack_to_qwen4exp import Arr
    kv += [('muse-spark.attention.sliding_window_pattern', 9, Arr(4, pattern))]
    tensors = [('token_embd.weight', 8, [8192, 202048]),
               ('output_norm.weight', 0, [8192]), ('output.weight', 8, [8192, 202048])]
    layer = [('attn_norm', 0, [8192]), ('attn_q', 14, [8192, 8192]),
             ('attn_k', 8, [8192, 1024]), ('attn_v', 8, [8192, 1024]),
             ('attn_output', 14, [8192, 8192]), ('attn_gate', 8, [8192, 8192]),
             ('attn_q_norm', 0, [64]), ('attn_k_norm', 0, [64]),
             ('attn_resid_alpha', 0, [8192]), ('attn_resid_beta', 0, [8192]),
             ('ffn_norm', 0, [8192]), ('post_ffw_norm', 0, [8192]),
             ('ffn_resid_alpha', 0, [8192]), ('ffn_resid_beta', 0, [8192]),
             ('ffn_gate_inp', 0, [8192, 256]), ('ffn_latent_down', 8, [8192, 4096]),
             ('ffn_latent_up', 8, [4096, 8192]), ('ffn_pre_exp_norm', 0, [4096]),
             ('ffn_post_exp_norm', 0, [4096])]
    for i in range(62):
        tensors += [(f'blk.{i}.{name}.weight', kind, dims) for name, kind, dims in layer]
        tensors.append((f'blk.{i}.exp_probs_b.bias', 0, [256]))
        for name in ('gate', 'up', 'down'):
            tensors.append((f'blk.{i}.ffn_{name}_exps.weight', 16, [4096, 4096, 256]))
    if bad == 'norm':
        tensors = [(n, 8 if n == 'blk.0.ffn_post_exp_norm.weight' else k, d) for n, k, d in tensors]
    if bad == 'pair':
        tensors = [(n, 8 if n == 'blk.0.ffn_up_exps.weight' else k, d) for n, k, d in tensors]
    if bad == 'missing': tensors = [t for t in tensors if t[0] != 'blk.0.attn_gate.weight']
    header = b'GGUF' + struct.pack('<IQQ', 3, len(tensors), len(kv))
    header += b''.join(kv_bytes(*entry) for entry in kv)
    offset = 0
    for name, kind, dims in tensors:
        offset = (offset + 31) // 32 * 32
        header += w_str(name) + struct.pack('<I', len(dims))
        header += struct.pack('<' + 'Q' * len(dims), *dims) + struct.pack('<IQ', kind, offset)
        offset += tensor_bytes(kind, dims)
    header += b'\0' * (-len(header) % 32)
    with path.open('wb') as fp:
        fp.write(header)
        fp.truncate(len(header) + offset)


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'sparse.gguf'
        for bad in (None, 'latent', 'pattern', 'norm', 'missing', 'pair'):
            fixture(path, bad)
            result = subprocess.run([sys.argv[1], str(path)], capture_output=True, text=True)
            if (result.returncode == 0) != (bad is None):
                raise AssertionError(f'{bad}: {result.stdout}\n{result.stderr}')
            print(f'Muse layout {bad or "valid"}: PASS')
