/* Muse Spark graph operations. Quantized dots are shared with the existing
 * native graphs; the model's GGUF weights need no new quantization format. */

struct muse_linear_args { uint in_dim, out_dim, rows, type; uint row_bytes; };

/* One SIMD group per output row. Keeping packed weights in the cache avoids
 * an unbounded FP16/BF16 dequantization allocation for a 256-expert bank. */
kernel void kernel_muse_linear(constant muse_linear_args &a,
    device const uchar *w, device const float *x, device float *out,
    uint3 group [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]]) {
    uint r = group.x, t = group.y;
    if (r >= a.out_dim || t >= a.rows) return;
    device const char *row = (device const char *)w + size_t(r) * a.row_bytes;
    float sum = ds4_quant_row_dot(row, x + size_t(t) * a.in_dim, a.type, a.in_dim, lane);
    if (lane == 0) out[size_t(t) * a.out_dim + r] = sum;
}

struct muse_norm_args { uint rows, width; float eps; };

kernel void kernel_muse_input_norm(constant muse_norm_args &a,
    device const float *x, device const float *w, device float *out,
    uint row [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]]) {
    if (row >= a.rows) return;
    size_t base = size_t(row) * a.width;
    float sum = 0;
    for (uint i = lane; i < a.width; i += 32) {
        float v = x[base + i];
        v *= w[i];
        sum += v * v;
    }
    float inv = metal::precise::rsqrt(simd_sum(sum) / a.width + a.eps);
    for (uint i = lane; i < a.width; i += 32) {
        float v = x[base + i];
        v *= w[i];
        v *= inv;
        out[base + i] = v;
    }
}

struct muse_attention_args {
    uint tokens, heads, kv_heads, dim, pos0, cap, window, rope;
    float rope_base;
};

/* Each lane owns a complete rotary pair, so Q can be rotated in place.
 * When building a ring from a longer batch, only its newest cap keys survive. */
kernel void kernel_muse_kv_store(constant muse_attention_args &a,
    device float *q, device half *kc, device half *vc,
    device const float *k, device const float *v,
    uint3 group [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]]) {
    uint h = group.x, t = group.y, d = 2u * lane;
    if (h >= a.heads || t >= a.tokens || d >= a.dim) return;
    float angle = a.rope ? float(a.pos0+t) * pow(a.rope_base,-float(d)/a.dim) : 0.f;
    float c = metal::precise::cos(angle), s = metal::precise::sin(angle);
    size_t b = (size_t(t)*a.heads+h)*a.dim+d;
    float x = q[b], y = q[b+1];
    q[b] = x*c-y*s; q[b+1] = x*s+y*c;
    if (h < a.kv_heads && a.tokens-t <= a.cap) {
        b = (size_t(t)*a.kv_heads+h)*a.dim+d;
        size_t to = (size_t((a.pos0+t)%a.cap)*a.kv_heads+h)*a.dim+d;
        x = k[b]; y = k[b+1];
        kc[to] = half(x*c-y*s); kc[to+1] = half(x*s+y*c);
        vc[to] = half(v[b]); vc[to+1] = half(v[b+1]);
    }
}

/* Split the visible keys over 32 SIMD groups per query head. Online softmax
 * keeps logits bounded without allocating a context-sized score matrix. */
kernel void kernel_muse_attention_split(constant muse_attention_args &a,
    device const float *q, device const half *kc, device const half *vc,
    device float *part,
    uint3 group [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]]) {
    uint h = group.x, split = group.y, t = group.z;
    uint end = a.pos0+t+1;
    uint first = a.window && end > a.window ? end-a.window : 0;
    uint chunk = (end-first+31)/32;
    uint lo = min(end, first+split*chunk), hi = min(end, lo+chunk);
    uint kh = h/(a.heads/a.kv_heads), d = 2u*lane;
    size_t qb = (size_t(t)*a.heads+h)*a.dim+d;
    float q0 = q[qb], q1 = q[qb+1];
    float m = -INFINITY, sum = 0.f, o0 = 0.f, o1 = 0.f;
    for (uint p = lo; p < hi; p++) {
        size_t b = (size_t(p%a.cap)*a.kv_heads+kh)*a.dim+d;
        float score = simd_sum(q0*float(kc[b])+q1*float(kc[b+1])) * rsqrt(float(a.dim));
        float next = max(m,score);
        float old = exp(m-next), add = exp(score-next);
        sum = sum*old+add;
        o0 = o0*old+add*float(vc[b]); o1 = o1*old+add*float(vc[b+1]);
        m = next;
    }
    size_t b = ((size_t(t)*a.heads+h)*32+split)*(a.dim+2);
    part[b+d] = o0; part[b+d+1] = o1;
    if (lane == 0) { part[b+a.dim] = m; part[b+a.dim+1] = sum; }
}

kernel void kernel_muse_attention_merge(constant muse_attention_args &a,
    device const float *part, device float *out,
    uint3 group [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]]) {
    uint h = group.x, t = group.y, d = 2u*lane;
    size_t base = (size_t(t)*a.heads+h)*32*(a.dim+2);
    float m = -INFINITY;
    for (uint i = 0; i < 32; i++) m = max(m,part[base+i*(a.dim+2)+a.dim]);
    float sum = 0.f, o0 = 0.f, o1 = 0.f;
    for (uint i = 0; i < 32; i++) {
        size_t b = base+i*(a.dim+2);
        float scale = exp(part[b+a.dim]-m);
        sum += scale*part[b+a.dim+1];
        o0 += scale*part[b+d]; o1 += scale*part[b+d+1];
    }
    size_t b = (size_t(t)*a.heads+h)*a.dim+d;
    out[b] = o0/sum; out[b+1] = o1/sum;
}

struct muse_pointwise_args { uint rows, width; float scale, cap; };

kernel void kernel_muse_residual(constant muse_pointwise_args &a,
    device const float *stream, device const float *branch,
    device const float *alpha, device const float *beta, device float *out,
    uint3 group [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]]) {
    uint i = group.x*32+lane, row = group.y;
    if (i >= a.width || row >= a.rows) return;
    size_t b = size_t(row)*a.width+i;
    out[b] = stream[b]*alpha[i]+branch[b]*beta[i];
}

kernel void kernel_muse_gate(constant muse_pointwise_args &a,
    device const float *x, device const float *gate, device float *out,
    uint group [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]]) {
    uint i = group*32+lane;
    if (i < a.width) out[i] = x[i]/(1.f+exp(-gate[i]));
}

kernel void kernel_muse_logits(constant muse_pointwise_args &a,
    device float *logits,
    uint group [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]]) {
    uint i = group*32+lane;
    if (i < a.width) logits[i] = a.cap*metal::precise::tanh(logits[i]*a.scale/a.cap);
}
