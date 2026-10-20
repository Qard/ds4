#ifndef DS4_MUSE_GPU_H
#define DS4_MUSE_GPU_H

#include "ds4_gpu.h"

/* GGUF graph activations use FP32; packed weights retain their native types. */
int ds4_gpu_muse_linear_tensor(ds4_gpu_tensor *out, const void *map, uint64_t size,
    uint64_t offset, uint32_t type, uint32_t in_dim, uint32_t out_dim,
    const ds4_gpu_tensor *x, uint32_t tokens);
int ds4_gpu_muse_input_norm_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
    const void *map, uint64_t size, uint64_t weight_offset,
    uint32_t rows, uint32_t width, float eps);

/* Normalized Q/K are FP32. KV is F16, token-major, with ring capacity cap.
 * For batched sliding-window prefill cap must hold window + tokens - 1. */
int ds4_gpu_muse_kv_store_tensor(ds4_gpu_tensor *q, ds4_gpu_tensor *kc, ds4_gpu_tensor *vc,
    const ds4_gpu_tensor *k, const ds4_gpu_tensor *v, uint32_t tokens,
    uint32_t heads, uint32_t kv_heads, uint32_t dim, uint32_t pos0,
    uint32_t cap, float rope_base, bool rope);
uint64_t ds4_gpu_muse_attention_scratch_bytes(uint32_t tokens, uint32_t heads, uint32_t dim);
int ds4_gpu_muse_attention_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *q,
    const ds4_gpu_tensor *kc, const ds4_gpu_tensor *vc, ds4_gpu_tensor *scratch,
    uint32_t tokens, uint32_t heads, uint32_t kv_heads, uint32_t dim,
    uint32_t pos0, uint32_t cap, uint32_t window);

int ds4_gpu_muse_residual_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *stream,
    const ds4_gpu_tensor *branch, const void *map, uint64_t size,
    uint64_t alpha_offset, uint64_t beta_offset, uint32_t rows, uint32_t width);
int ds4_gpu_muse_gate_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
    const ds4_gpu_tensor *gate, uint32_t n);
int ds4_gpu_muse_logits_tensor(ds4_gpu_tensor *logits, uint32_t n, float scale, float cap);

#endif
