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

/* Visit each lane's values in the original ascending order, while reusing
 * the packed bytes shared by four quarters of a Q6_K half-block. Explicit
 * FMAs keep the serial FP32 accumulation intact when unrolling the quarters. */
kernel void kernel_muse_linear_q6(constant muse_linear_args &a,
    device const uchar *w,device const float *x,device float *out,
    uint3 group [[threadgroup_position_in_grid]],ushort lane [[thread_index_in_simdgroup]]) {
    uint r=group.x,t=group.y;
    if(r>=a.out_dim||t>=a.rows) return;
    device const block_q6_K *blocks=(device const block_q6_K *)(w+size_t(r)*a.row_bytes);
    device const float *input=x+size_t(t)*a.in_dim;
    float acc=0;
    for(uint b=0;b<a.in_dim/256;b++) {
        device const block_q6_K *block=blocks+b;
        float d=float(block->d);
#pragma unroll
        for(uint h=0;h<2;h++) {
            uint lo0=block->ql[h*64+lane],lo1=block->ql[h*64+32+lane];
            uint hi=block->qh[h*32+lane],s=h*8+lane/16;
            device const float *v=input+b*256+h*128+lane;
            float q0=float(int((lo0&15u)|((hi&3u)<<4))-32);
            float q1=float(int((lo1&15u)|(((hi>>2)&3u)<<4))-32);
            float q2=float(int((lo0>>4)|(((hi>>4)&3u)<<4))-32);
            float q3=float(int((lo1>>4)|(((hi>>6)&3u)<<4))-32);
            acc=fma((d*float(block->scales[s]))*q0,v[0],acc);
            acc=fma((d*float(block->scales[s+2]))*q1,v[32],acc);
            acc=fma((d*float(block->scales[s+4]))*q2,v[64],acc);
            acc=fma((d*float(block->scales[s+6]))*q3,v[96],acc);
        }
    }
    float sum=simd_sum(acc);
    if(lane==0) out[size_t(t)*a.out_dim+r]=sum;
}

/* Preserve the serial FMA sequence while exposing four blocks to the
 * scheduler. SIMD groups own independent rows; prefill can pack them. */
kernel void kernel_muse_linear_q6_tiled(constant muse_linear_args &a,
    device const uchar *w,device const float *x,device float *out,
    uint3 group [[threadgroup_position_in_grid]],ushort lane [[thread_index_in_simdgroup]],
    ushort simd [[simdgroup_index_in_threadgroup]],uint3 threads [[threads_per_threadgroup]]) {
    uint r=group.x*threads.y+simd,t=group.y;
    if(r>=a.out_dim||t>=a.rows) return;
    device const block_q6_K *blocks=(device const block_q6_K *)(w+size_t(r)*a.row_bytes);
    device const float *input=x+size_t(t)*a.in_dim;
    float acc=0;
#pragma unroll(4)
    for(uint b=0;b<a.in_dim/256;b++) {
        device const block_q6_K *block=blocks+b;
        float d=float(block->d);
#pragma unroll
        for(uint h=0;h<2;h++) {
            uint lo0=block->ql[h*64+lane],lo1=block->ql[h*64+32+lane];
            uint hi=block->qh[h*32+lane],s=h*8+lane/16;
            device const float *v=input+b*256+h*128+lane;
            float q0=float(int((lo0&15u)|((hi&3u)<<4))-32);
            float q1=float(int((lo1&15u)|(((hi>>2)&3u)<<4))-32);
            float q2=float(int((lo0>>4)|(((hi>>4)&3u)<<4))-32);
            float q3=float(int((lo1>>4)|(((hi>>6)&3u)<<4))-32);
            acc=fma((d*float(block->scales[s]))*q0,v[0],acc);
            acc=fma((d*float(block->scales[s+2]))*q1,v[32],acc);
            acc=fma((d*float(block->scales[s+4]))*q2,v[64],acc);
            acc=fma((d*float(block->scales[s+6]))*q3,v[96],acc);
        }
    }
    float sum=simd_sum(acc);
    if(lane==0) out[size_t(t)*a.out_dim+r]=sum;
}

/* Four RHS rows share packed bytes and dequantized FP32 weights. Each RHS
 * retains the same lane ownership and serial FMAs as one-token decode. */
kernel void kernel_muse_linear_q6_batch4(constant muse_linear_args &a,
    device const uchar *w,device const float *x,device float *out,
    uint3 group [[threadgroup_position_in_grid]],ushort lane [[thread_index_in_simdgroup]],
    ushort simd [[simdgroup_index_in_threadgroup]],uint3 threads [[threads_per_threadgroup]]) {
    uint row=group.x*threads.y+simd,t=group.y*4;
    if(row>=a.out_dim||t>=a.rows) return;
    device const block_q6_K *blocks=(device const block_q6_K *)(w+size_t(row)*a.row_bytes);
    float acc[4]={0,0,0,0};
    for(uint b=0;b<a.in_dim/256;b++) {
        device const block_q6_K *block=blocks+b;float d=float(block->d);
#pragma unroll
        for(uint h=0;h<2;h++) {
            uint lo0=block->ql[h*64+lane],lo1=block->ql[h*64+32+lane];
            uint hi=block->qh[h*32+lane],sc=h*8+lane/16;
            float q0=float(int((lo0&15u)|((hi&3u)<<4))-32);
            float q1=float(int((lo1&15u)|(((hi>>2)&3u)<<4))-32);
            float q2=float(int((lo0>>4)|(((hi>>4)&3u)<<4))-32);
            float q3=float(int((lo1>>4)|(((hi>>6)&3u)<<4))-32);
            float w0=(d*float(block->scales[sc]))*q0,w1=(d*float(block->scales[sc+2]))*q1;
            float w2=(d*float(block->scales[sc+4]))*q2,w3=(d*float(block->scales[sc+6]))*q3;
#pragma unroll
            for(uint r=0;r<4;r++) if(t+r<a.rows) {
                device const float *v=x+size_t(t+r)*a.in_dim+b*256+h*128+lane;
                acc[r]=fma(w0,v[0],acc[r]);acc[r]=fma(w1,v[32],acc[r]);
                acc[r]=fma(w2,v[64],acc[r]);acc[r]=fma(w3,v[96],acc[r]);
            }
        }
    }
#pragma unroll
    for(uint r=0;r<4;r++) {
        float sum=simd_sum(acc[r]);
        if(lane==0&&t+r<a.rows) out[size_t(t+r)*a.out_dim+row]=sum;
    }
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

/* Prefetch four strided values before consuming the serial FMA chain. The
 * per-lane order and SIMD reduction remain identical to the scalar loop. */
kernel void kernel_muse_input_norm_unroll(constant muse_norm_args &a,
    device const float *x,device const float *w,device float *out,
    uint row [[threadgroup_position_in_grid]],ushort lane [[thread_index_in_simdgroup]]) {
    if(row>=a.rows) return;
    size_t base=size_t(row)*a.width;
    float sum=0;uint i=lane;
    for(;i+96<a.width;i+=128) {
        float v0=x[base+i]*w[i],v1=x[base+i+32]*w[i+32];
        float v2=x[base+i+64]*w[i+64],v3=x[base+i+96]*w[i+96];
        sum=fma(v0,v0,sum);sum=fma(v1,v1,sum);
        sum=fma(v2,v2,sum);sum=fma(v3,v3,sum);
    }
    for(;i<a.width;i+=32) {float v=x[base+i]*w[i];sum=fma(v,v,sum);}
    float inv=metal::precise::rsqrt(simd_sum(sum)/a.width+a.eps);
#pragma unroll(4)
    for(uint j=lane;j<a.width;j+=32) {
        float v=x[base+j];v*=w[j];v*=inv;out[base+j]=v;
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

struct muse_expert_args {
    uint tokens, slots, latent, ff, gate_type, down_type, gate_row, down_row, total;
};

/* The existing native SSD cache owns these GPU addresses and their lifetime.
 * Routing weights intentionally apply only after the per-expert post norm. */
kernel void kernel_muse_expert_mid(constant muse_expert_args &a,
    device const uint64_t *gates, device const uint64_t *ups,
    device const int *selected, device const float *x, device float *mid,
    uint3 group [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]]) {
    uint row = group.x, slot = group.y, tok = group.z;
    int expert = selected[size_t(tok)*a.slots+slot];
    if (expert < 0 || uint(expert) >= a.total) return;
    device const char *gate = reinterpret_cast<device const char *>(gates[expert])+size_t(row)*a.gate_row;
    device const char *up = reinterpret_cast<device const char *>(ups[expert])+size_t(row)*a.gate_row;
    device const float *input = x+size_t(tok)*a.latent;
    float g = ds4_quant_row_dot(gate,input,a.gate_type,a.latent,lane);
    float u = ds4_quant_row_dot(up,input,a.gate_type,a.latent,lane);
    if (lane == 0) mid[(size_t(tok)*a.slots+slot)*a.ff+row] = g/(1.f+exp(-g))*u;
}

/* Gate and up share input loads, but retain independent original serial
 * FP32 chains and SIMD sums. Existing MoE pair kernels use a different
 * per-lane layout, so retain the generic row-dot ordering here. */
static inline float2 muse_iq2_pair_dot(device const char *gate,device const char *up,
    device const float *input,uint width,ushort lane) {
    uint sub=lane/4,j=lane%4;float2 acc=0.f;
#pragma unroll(2)
    for(uint b=0;b<width/256;b++) {
        device const uchar *gb=(device const uchar *)(gate+size_t(b)*66);
        device const uchar *ub=(device const uchar *)(up+size_t(b)*66);
        device const ushort *gq=(device const ushort *)(gb+2)+4*sub;
        device const ushort *uq=(device const ushort *)(ub+2)+4*sub;
        uint gg=uint(gq[0])|(uint(gq[1])<<16),gs=uint(gq[2])|(uint(gq[3])<<16);
        uint ug=uint(uq[0])|(uint(uq[1])<<16),us=uint(uq[2])|(uint(uq[3])<<16);
        float gd=float(*(device const half *)gb)*(0.5f+float(gs>>28))*0.25f;
        float ud=float(*(device const half *)ub)*(0.5f+float(us>>28))*0.25f;
        constant const uchar *gv=(constant const uchar *)(ds4_metal_iq2xxs_grid+((gg>>(8*j))&255u));
        constant const uchar *uv=(constant const uchar *)(ds4_metal_iq2xxs_grid+((ug>>(8*j))&255u));
        uint gsign=ds4_metal_ksigns_iq2xs[(gs>>(7*j))&127u];
        uint usign=ds4_metal_ksigns_iq2xs[(us>>(7*j))&127u];
        device const float *x=input+b*256+sub*32+j*8;float2 part=0.f;
#pragma unroll
        for(uint i=0;i<8;i++) {
            float value=x[i];
            part.x=fma(float(gv[i]),((gsign>>i)&1u)?-value:value,part.x);
            part.y=fma(float(uv[i]),((usign>>i)&1u)?-value:value,part.y);
        }
        acc.x=fma(gd,part.x,acc.x);acc.y=fma(ud,part.y,acc.y);
    }
    return float2(simd_sum(acc.x),simd_sum(acc.y));
}

kernel void kernel_muse_expert_mid_iq2_pair(constant muse_expert_args &a,
    device const uint64_t *gates, device const uint64_t *ups,
    device const int *selected, device const float *x, device float *mid,
    uint3 group [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]]) {
    uint row = group.x, slot = group.y, tok = group.z;
    int expert = selected[size_t(tok)*a.slots+slot];
    if (expert < 0 || uint(expert) >= a.total) return;
    device const char *gate = reinterpret_cast<device const char *>(gates[expert])+size_t(row)*a.gate_row;
    device const char *up = reinterpret_cast<device const char *>(ups[expert])+size_t(row)*a.gate_row;
    device const float *input = x+size_t(tok)*a.latent;
    float2 pair=muse_iq2_pair_dot(gate,up,input,a.latent,lane);
    float g=pair.x,u=pair.y;
    if (lane == 0) mid[(size_t(tok)*a.slots+slot)*a.ff+row] = g/(1.f+exp(-g))*u;
}

kernel void kernel_muse_expert_down(constant muse_expert_args &a,
    device const uint64_t *downs, device const int *selected,
    device const float *mid, device float *part,
    uint3 group [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]]) {
    uint row = group.x, slot = group.y, tok = group.z;
    int expert = selected[size_t(tok)*a.slots+slot];
    if (expert < 0 || uint(expert) >= a.total) return;
    device const char *down = reinterpret_cast<device const char *>(downs[expert])+size_t(row)*a.down_row;
    float v = ds4_quant_row_dot(down,mid+(size_t(tok)*a.slots+slot)*a.ff,a.down_type,a.ff,lane);
    if (lane == 0) part[(size_t(tok)*a.slots+slot)*a.latent+row] = v;
}

/* Pair lists are built by the shared Qwen operator. Their ordering is free:
 * each RHS keeps the original per-lane IQ2 FMAs and writes its own flat slot. */
struct muse_expert_batch_args {
    muse_expert_args base;
    uint list_cap,tiles,experts[512];
};

template <ushort N>
static inline float4 muse_iq2_batch_dot(device const char *row,
    device const float *x0,device const float *x1,device const float *x2,device const float *x3,
    uint width,ushort lane) {
    device const float *inputs[4]={x0,x1,x2,x3};
    float4 acc=0.f;
    uint sub=lane/4,j=lane%4;
    for(uint b=0;b<width/256;b++) {
        device const uchar *block=(device const uchar *)(row+size_t(b)*66);
        device const ushort *q=(device const ushort *)(block+2)+4*sub;
        uint grids=uint(q[0])|(uint(q[1])<<16),scales=uint(q[2])|(uint(q[3])<<16);
        float d=float(*(device const half *)block)*(0.5f+float(scales>>28))*0.25f;
        constant const uchar *grid=(constant const uchar *)(ds4_metal_iq2xxs_grid+((grids>>(8*j))&255u));
        uint signs=ds4_metal_ksigns_iq2xs[(scales>>(7*j))&127u];
        uint base=b*256+sub*32+j*8;float4 part=0.f;
#pragma unroll
        for(uint i=0;i<8;i++) {
            float value=float(grid[i]);bool neg=((signs>>i)&1u)!=0;
#pragma unroll
            for(uint r=0;r<N;r++) {
                float v=inputs[r][base+i];part[r]=fma(value,neg?-v:v,part[r]);
            }
        }
#pragma unroll
        for(uint r=0;r<N;r++) acc[r]=fma(d,part[r],acc[r]);
    }
    return float4(simd_sum(acc.x),simd_sum(acc.y),simd_sum(acc.z),simd_sum(acc.w));
}

static inline float4 muse_iq2_batch_select(device const char *row,
    device const float *x0,device const float *x1,device const float *x2,device const float *x3,
    uint width,ushort lane,uint n) {
    if(n==1) return float4(ds4_quant_row_dot(row,x0,16,width,lane),0.f,0.f,0.f);
    if(n==2) return muse_iq2_batch_dot<2>(row,x0,x1,x2,x3,width,lane);
    if(n==3) return muse_iq2_batch_dot<3>(row,x0,x1,x2,x3,width,lane);
    return muse_iq2_batch_dot<4>(row,x0,x1,x2,x3,width,lane);
}

kernel void kernel_muse_expert_mid_iq2_batch(constant muse_expert_batch_args &b,
    device const uint64_t *gates,device const uint64_t *ups,device const float *x,device float *mid,
    device const int *lists,device const int *counts,
    uint3 group [[threadgroup_position_in_grid]],ushort lane [[thread_index_in_simdgroup]],
    ushort simd [[simdgroup_index_in_threadgroup]],uint3 threads [[threads_per_threadgroup]]) {
    constant muse_expert_args &a=b.base;
    uint row=group.x*threads.y+simd,expert=b.experts[group.y],count=uint(counts[expert]);
    uint first=group.z*4;
    if(first>=count || row>=a.ff) return;
    device const char *gate=reinterpret_cast<device const char *>(gates[expert])+size_t(row)*a.gate_row;
    device const char *up=reinterpret_cast<device const char *>(ups[expert])+size_t(row)*a.gate_row;
    for(;first<count;first+=b.tiles*4) {
        uint n=min(4u,count-first);int pair[4];
#pragma unroll
        for(uint r=0;r<4;r++) pair[r]=lists[size_t(expert)*b.list_cap+first+(r<n?r:0)];
        device const float *x0=x+size_t(uint(pair[0])/a.slots)*a.latent;
        device const float *x1=x+size_t(uint(pair[1])/a.slots)*a.latent;
        device const float *x2=x+size_t(uint(pair[2])/a.slots)*a.latent;
        device const float *x3=x+size_t(uint(pair[3])/a.slots)*a.latent;
        float4 g=muse_iq2_batch_select(gate,x0,x1,x2,x3,a.latent,lane,n);
        float4 u=muse_iq2_batch_select(up,x0,x1,x2,x3,a.latent,lane,n);
        if(lane==0) {
#pragma unroll
            for(uint r=0;r<4;r++) if(r<n)
                mid[size_t(uint(pair[r]))*a.ff+row]=g[r]/(1.f+exp(-g[r]))*u[r];
        }
    }
}

kernel void kernel_muse_expert_down_iq2_batch(constant muse_expert_batch_args &b,
    device const uint64_t *downs,device const float *mid,device float *part,
    device const int *lists,device const int *counts,
    uint3 group [[threadgroup_position_in_grid]],ushort lane [[thread_index_in_simdgroup]],
    ushort simd [[simdgroup_index_in_threadgroup]],uint3 threads [[threads_per_threadgroup]]) {
    constant muse_expert_args &a=b.base;
    uint row=group.x*threads.y+simd,expert=b.experts[group.y],count=uint(counts[expert]);
    uint first=group.z*4;
    if(first>=count || row>=a.latent) return;
    device const char *down=reinterpret_cast<device const char *>(downs[expert])+size_t(row)*a.down_row;
    for(;first<count;first+=b.tiles*4) {
        uint n=min(4u,count-first);int pair[4];
#pragma unroll
        for(uint r=0;r<4;r++) pair[r]=lists[size_t(expert)*b.list_cap+first+(r<n?r:0)];
        float4 v=muse_iq2_batch_select(down,mid+size_t(uint(pair[0]))*a.ff,
            mid+size_t(uint(pair[1]))*a.ff,mid+size_t(uint(pair[2]))*a.ff,
            mid+size_t(uint(pair[3]))*a.ff,a.ff,lane,n);
        if(lane==0) {
#pragma unroll
            for(uint r=0;r<4;r++) if(r<n) part[size_t(uint(pair[r]))*a.latent+row]=v[r];
        }
    }
}
