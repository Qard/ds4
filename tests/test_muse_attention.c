#include "ds4_muse_gpu.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static void check_rows(unsigned T, unsigned cap, unsigned window, bool rope, unsigned tail_rows) {
    enum { H = 4, KV = 2, D = 64 };
    size_t qbytes=(size_t)T*H*D*sizeof(float),kbytes=(size_t)T*KV*D*sizeof(float);
    float *q=malloc(qbytes),*qr=malloc(qbytes),*k=malloc(kbytes),*v=malloc(kbytes),*kr=malloc(kbytes);
    assert(q && qr && k && v && kr);
    for (unsigned i = 0; i < T*H*D; i++) q[i] = ((int)((i*17)%23)-11)*.0625f;
    for (unsigned i = 0; i < T*KV*D; i++) {
        k[i] = ((int)((i*7)%19)-9)*.0625f;
        v[i] = ((int)((i*13)%29)-14)*.125f;
    }
    if (tail_rows && !window && !rope) {
        /* An early key must remain visible thousands of tokens later. A
         * mistaken sliding window would return zero, far outside tolerance. */
        for(unsigned i=0;i<T*H*D;i++) q[i]=0;
        for(unsigned i=0;i<T*KV*D;i++) k[i]=v[i]=0;
        for(unsigned t=T-tail_rows;t<T;t++) for(unsigned h=0;h<H;h++) q[(t*H+h)*D]=8;
        for(unsigned h=0;h<KV;h++) {
            k[h*D]=8;
            for(unsigned d=0;d<D;d++) v[h*D+d]=.03125f*(d+1)*(h+1);
        }
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
    ds4_gpu_tensor *gq = ds4_gpu_tensor_alloc(qbytes);
    ds4_gpu_tensor *gk = ds4_gpu_tensor_alloc(kbytes);
    ds4_gpu_tensor *gv = ds4_gpu_tensor_alloc(kbytes);
    ds4_gpu_tensor *kc = ds4_gpu_tensor_alloc(cap*KV*D*2);
    ds4_gpu_tensor *vc = ds4_gpu_tensor_alloc(cap*KV*D*2);
    assert(gq && gk && gv && kc && vc);
    assert(ds4_gpu_tensor_write(gq, 0, q, qbytes));
    assert(ds4_gpu_tensor_write(gk, 0, k, kbytes));
    assert(ds4_gpu_tensor_write(gv, 0, v, kbytes));
    assert(ds4_gpu_begin_commands());
    assert(ds4_gpu_muse_kv_store_tensor(gq, kc, vc, gk, gv, T,H,KV,D,0,cap,500000.f,rope));
    unsigned start = tail_rows ? T-tail_rows : cap < T ? T-2 : 0, rows = T-start;
    ds4_gpu_tensor *query = ds4_gpu_tensor_view(gq,start*H*D*4,rows*H*D*4);
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(rows*H*D*4);
    ds4_gpu_tensor *part = ds4_gpu_tensor_alloc(ds4_gpu_muse_attention_scratch_bytes(rows,H,D));
    assert(query && out && part);
    assert(ds4_gpu_muse_attention_tensor(out,query,kc,vc,part,rows,H,KV,D,start,cap,window));
    assert(ds4_gpu_end_commands());
    float *actual=malloc((size_t)rows*H*D*sizeof(float));assert(actual);
    assert(ds4_gpu_synchronize() && ds4_gpu_tensor_read(out,0,actual,rows*H*D*4));
    double max_error=0;
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
            double error=fabs(actual[((t-start)*H+h)*D+d]-expected);
            if(error>max_error) max_error=error;
            assert(error<3e-5);
        }
    }
    if(tail_rows) printf("Muse %u-token %s tail parity: max error %.9g PASS\n",T,window?"sliding-window":"global NoPE",max_error);
    assert(!ds4_gpu_muse_attention_tensor(out,query,kc,vc,part,rows,H,KV,D,start,1,window));
    ds4_gpu_tensor_free(part); ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(query);
    ds4_gpu_tensor_free(vc); ds4_gpu_tensor_free(kc); ds4_gpu_tensor_free(gv);
    ds4_gpu_tensor_free(gk); ds4_gpu_tensor_free(gq);
    free(actual);free(kr);free(qr);free(v);free(k);free(q);
}

static void check(unsigned T,unsigned cap,unsigned window,bool rope) {check_rows(T,cap,window,rope,0);}

int main(void) {
    assert(ds4_gpu_init());
    check(10,16,0,false); check(10,16,4,true); check(10,7,4,true);
    check(70,128,0,false); check(70,128,35,true); check(70,39,35,true);
    /* Compare only the newest queries, but include every visible key in the
     * independent FP64 reference. Keep 8K tensors on the heap. */
    check_rows(8193,16384,0,false,2);
    check_rows(8193,2560,2048,true,2);
    ds4_gpu_cleanup();
    puts("Muse causal GQA, RoPE/NoPE, and sliding KV ring: PASS");
}
