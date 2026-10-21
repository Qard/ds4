#ifndef DS4_MUSE_GPU_H
#define DS4_MUSE_GPU_H

#include "ds4_gpu.h"

/* GGUF graph activations use FP32; packed weights retain their native types. */
int ds4_gpu_muse_linear_tensor(ds4_gpu_tensor *out, const void *map, uint64_t size,
    uint64_t offset, uint32_t type, uint32_t in_dim, uint32_t out_dim,
    const ds4_gpu_tensor *x, uint32_t tokens);
/* One to four independent projections share an input. Output ranges must be
 * disjoint from the input and each other; nonoverlapping buffer views are OK.
 * Any concurrent section it starts ends before returning, including on failure.
 * The caller retains ownership of an active command batch. */
int ds4_gpu_muse_linears_tensor(ds4_gpu_tensor *const *out,
    const void *map, uint64_t size, const uint64_t *offsets, const uint32_t *types,
    const uint32_t *widths, uint32_t count, uint32_t in_dim,
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

/* Raw expert outputs [tokens][slots][latent]; Muse applies its input-scaled
 * per-expert norm before the existing weighted reduction. Gate/up share type. */
/* Begin the existing bounded pread work after routing, before latent projection.
 * Every begun load is joined before expert dispatch or graph failure returns. */
int ds4_gpu_muse_prefetch_experts_tensor(const ds4_gpu_stream_expert_table *table,
    const ds4_gpu_tensor *selected,uint32_t n_selected,uint64_t routing_event);
int ds4_gpu_muse_prefetch_finish(void);
/* routing_event is zero unless this selection has been read back through
 * prefetch_experts_tensor. A current completed event permits CPU address
 * preparation while the tracked latent producer remains in flight. */
int ds4_gpu_muse_experts_tensor(ds4_gpu_tensor *mid, ds4_gpu_tensor *part,
    const ds4_gpu_tensor *x, const ds4_gpu_tensor *selected,
    const void *map, uint64_t size, uint64_t gate_offset, uint64_t up_offset,
    uint64_t down_offset, uint32_t gate_type, uint32_t down_type,
    uint32_t tokens, uint32_t total, uint32_t slots, uint32_t latent,
    uint32_t ff, uint32_t layer,ds4_gpu_tensor *lists,ds4_gpu_tensor *counts,
    uint64_t routing_event);

#endif
