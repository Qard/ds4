#include "ds4_muse_gpu.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static void check(unsigned T, unsigned cap, unsigned window, bool rope) {
    enum { H = 4, KV = 2, D = 64 };
    float q[T*H*D], k[T*KV*D], v[T*KV*D], qr[T*H*D], kr[T*KV*D];
    for (unsigned i = 0; i < T*H*D; i++) q[i] = ((int)((i*17)%23)-11)*.0625f;
    for (unsigned i = 0; i < T*KV*D; i++) {
        k[i] = ((int)((i*7)%19)-9)*.0625f;
        v[i] = ((int)((i*13)%29)-14)*.125f;
    }
    for (unsigned t = 0; t < T; t++) for (unsigned h = 0; h < H; h++)
    for (unsigned d = 0; d < D; d += 2) {
        double angle = rope ? t * pow(500000., -(double)d/D) : 0;
        unsigned b = (t*H+h)*D+d;
        qr[b] = q[b]*cos(angle)-q[b+1]*sin(angle);
        qr[b+1] = q[b]*sin(angle)+q[b+1]*cos(angle);
        if (h < KV) {
            b = (t*KV+h)*D+d;
            kr[b] = (float)(_Float16)(k[b]*cos(angle)-k[b+1]*sin(angle));
            kr[b+1] = (float)(_Float16)(k[b]*sin(angle)+k[b+1]*cos(angle));
        }
    }
    ds4_gpu_tensor *gq = ds4_gpu_tensor_alloc(sizeof(q));
    ds4_gpu_tensor *gk = ds4_gpu_tensor_alloc(sizeof(k));
    ds4_gpu_tensor *gv = ds4_gpu_tensor_alloc(sizeof(v));
    ds4_gpu_tensor *kc = ds4_gpu_tensor_alloc(cap*KV*D*2);
    ds4_gpu_tensor *vc = ds4_gpu_tensor_alloc(cap*KV*D*2);
    assert(gq && gk && gv && kc && vc);
    assert(ds4_gpu_tensor_write(gq, 0, q, sizeof(q)));
    assert(ds4_gpu_tensor_write(gk, 0, k, sizeof(k)));
    assert(ds4_gpu_tensor_write(gv, 0, v, sizeof(v)));
    assert(ds4_gpu_begin_commands());
    assert(ds4_gpu_muse_kv_store_tensor(gq, kc, vc, gk, gv, T,H,KV,D,0,cap,500000.f,rope));
    unsigned start = cap < T ? T-2 : 0, rows = T-start;
    ds4_gpu_tensor *query = ds4_gpu_tensor_view(gq,start*H*D*4,rows*H*D*4);
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(rows*H*D*4);
    ds4_gpu_tensor *part = ds4_gpu_tensor_alloc(ds4_gpu_muse_attention_scratch_bytes(rows,H,D));
    assert(query && out && part);
    assert(ds4_gpu_muse_attention_tensor(out,query,kc,vc,part,rows,H,KV,D,start,cap,window));
    assert(ds4_gpu_end_commands());
    float actual[T*H*D];
    assert(ds4_gpu_synchronize() && ds4_gpu_tensor_read(out,0,actual,rows*H*D*4));
    for (unsigned t = start; t < T; t++) for (unsigned h = 0; h < H; h++) {
        unsigned lo = window && t+1>window ? t+1-window : 0;
        double score[T], sum=0;
        for (unsigned p=lo;p<=t;p++) {
            score[p]=0;
            for (unsigned d=0;d<D;d++) score[p] += qr[(t*H+h)*D+d]*kr[(p*KV+h/(H/KV))*D+d]/sqrt(D);
            score[p]=exp(score[p]); sum+=score[p];
        }
        for (unsigned d=0;d<D;d++) {
            double expected=0;
            for (unsigned p=lo;p<=t;p++) expected += score[p]/sum*v[(p*KV+h/(H/KV))*D+d];
            assert(fabs(actual[((t-start)*H+h)*D+d]-expected)<3e-5);
        }
    }
    assert(!ds4_gpu_muse_attention_tensor(out,query,kc,vc,part,rows,H,KV,D,start,1,window));
    ds4_gpu_tensor_free(part); ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(query);
    ds4_gpu_tensor_free(vc); ds4_gpu_tensor_free(kc); ds4_gpu_tensor_free(gv);
    ds4_gpu_tensor_free(gk); ds4_gpu_tensor_free(gq);
}

int main(void) {
    assert(ds4_gpu_init());
    check(10,16,0,false); check(10,16,4,true); check(10,7,4,true);
    check(70,128,0,false); check(70,128,35,true); check(70,39,35,true);
    ds4_gpu_cleanup();
    puts("Muse causal GQA, RoPE/NoPE, and sliding KV ring: PASS");
}
